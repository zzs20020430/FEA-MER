import torch
import gc
import threading
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import subprocess
import os
import ctypes
from ctypes import c_void_p, c_size_t, c_int, c_char_p

# 全局变量用于保持对预分配tensor的引用
_global_reserved_tensors = {}

class GPUMemoryPool:
    """
    基于伙伴算法的GPU显存池管理器
    预先分配所有可用的GPU显存，然后按需分配给PyTorch操作
    """
    def __init__(self, 
                 devices: List[int] = None, 
                 allocation_ratio: float = 0.95,
                 min_block_size: int = 1024 * 1024,  # 1MB
                 enable_logging: bool = False):
        """
        初始化显存池管理器
        
        Args:
            devices: GPU设备ID列表，None表示使用所有可用设备
            allocation_ratio: 每个设备分配的显存比例(0.0-1.0)
            min_block_size: 最小分配块大小(字节)
            enable_logging: 是否启用日志记录
        """
        self.allocation_ratio = allocation_ratio
        self.min_block_size = min_block_size
        self.enable_logging = enable_logging
        self.memory_pools = {}
        self.memory_blocks = {}
        self.allocation_map = {}
        self.lock = threading.RLock()
        self._reserved_tensors = {}  # 用于保持对预分配tensor的引用
        
        # 设置日志
        self.logger = logging.getLogger("GPUMemoryPool")
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
        
        # 获取CUDA_VISIBLE_DEVICES环境变量
        self.visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if self.visible_devices:
            try:
                # 解析环境变量中的GPU ID
                self.physical_gpus = [int(id.strip()) for id in self.visible_devices.split(',')]
                # 创建物理GPU ID到逻辑GPU ID的映射
                self.physical_to_logical = {physical: logical for logical, physical in enumerate(self.physical_gpus)}
                self.logical_to_physical = {logical: physical for physical, logical in self.physical_to_logical.items()}
            except:
                self.physical_gpus = list(range(torch.cuda.device_count()))
                self.physical_to_logical = {i: i for i in self.physical_gpus}
                self.logical_to_physical = {i: i for i in self.physical_gpus}
        else:
            self.physical_gpus = list(range(torch.cuda.device_count()))
            self.physical_to_logical = {i: i for i in self.physical_gpus}
            self.logical_to_physical = {i: i for i in self.physical_gpus}
        
        # 确定要管理的设备
        if devices is None:
            devices = self.physical_gpus
        self.devices = devices
        
        if self.enable_logging:
            self.logger.info(f"物理GPU设备: {self.devices}")
            self.logger.info(f"GPU ID映射: {self.physical_to_logical}")
        
        # 初始化每个设备的内存池
        for device in self.devices:
            self._init_device_pool(device)
        
        # 设置PyTorch的内存分配器配置
        self._configure_pytorch_allocator()
        
        if self.enable_logging:
            self.logger.info(f"GPU内存池初始化完成，管理设备: {self.devices}")
    
    def _configure_pytorch_allocator(self):
        """配置PyTorch的内存分配器"""
        # 设置环境变量
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
            'max_split_size_mb:512,'
            'garbage_collection_threshold:0.8'
        )
        
        # 设置CUDA内存分配器
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            for device in self.devices:
                logical_device = self.physical_to_logical[device]
                torch.cuda.set_per_process_memory_fraction(1.0, logical_device)
    
    def _allocate_from_pool(self, size: int, device: int = None) -> Optional[torch.Tensor]:
        """
        从内存池中分配指定大小的内存
        
        Args:
            size: 需要分配的内存大小(字节)
            device: GPU设备ID
            
        Returns:
            torch.Tensor: 分配的内存tensor，如果分配失败则返回None
        """
        if device is None:
            device = torch.cuda.current_device()
        
        # 将逻辑GPU ID转换为物理GPU ID
        physical_device = self.logical_to_physical.get(device, device)
        
        with self.lock:
            # 查找合适的空闲块
            block = self._find_free_block(size, physical_device)
            if block is None:
                return None
            
            # 分配内存块
            tensor = self._allocate_block(block, size, physical_device)
            return tensor
    
    def _find_free_block(self, size: int, device: int) -> Optional[Tuple[int, int]]:
        """查找足够大的空闲内存块"""
        # 将size向上取整到最小块大小的倍数
        aligned_size = ((size + self.min_block_size - 1) // self.min_block_size) * self.min_block_size
        
        for i, (start, block_size) in enumerate(self.memory_blocks[device]):
            if block_size >= aligned_size:
                return (start, block_size)
        return None
    
    def _allocate_block(self, block: Tuple[int, int], size: int, device: int) -> torch.Tensor:
        """分配内存块并更新内存池状态"""
        start, block_size = block
        
        # 创建tensor
        tensor = torch.empty(size, dtype=torch.uint8, device=f"cuda:{self.physical_to_logical[device]}")
        
        # 更新内存块状态
        if block_size > size + self.min_block_size:
            # 分割块
            self.memory_blocks[device][i] = (start, size)
            self.memory_blocks[device].insert(i + 1, (start + size, block_size - size))
        else:
            # 使用整个块
            self.memory_blocks[device].pop(i)
        
        # 记录分配
        self.allocation_map[device][start] = (size, tensor)
        
        return tensor
    
    def _free_block(self, start: int, device: int):
        """释放内存块并合并相邻的空闲块"""
        if start not in self.allocation_map[device]:
            return
        
        size, tensor = self.allocation_map[device][start]
        del self.allocation_map[device][start]
        
        # 添加新的空闲块
        self.memory_blocks[device].append((start, size))
        self.memory_blocks[device].sort()
        
        # 合并相邻的空闲块
        i = 0
        while i < len(self.memory_blocks[device]) - 1:
            curr_start, curr_size = self.memory_blocks[device][i]
            next_start, next_size = self.memory_blocks[device][i + 1]
            
            if curr_start + curr_size == next_start:
                # 合并块
                self.memory_blocks[device][i] = (curr_start, curr_size + next_size)
                self.memory_blocks[device].pop(i + 1)
            else:
                i += 1
    
    def get_free_memory(self, device: int = None) -> int:
        """获取设备上可用显存总量"""
        if device is None:
            device = torch.cuda.current_device()
        
        # 将逻辑GPU ID转换为物理GPU ID
        physical_device = self.logical_to_physical.get(device, device)
        
        if physical_device not in self.devices:
            return 0
        
        with self.lock:
            return sum(size for _, size in self.memory_blocks[physical_device])
    
    def get_allocated_memory(self, device: int = None) -> int:
        """获取设备上已分配显存总量"""
        if device is None:
            device = torch.cuda.current_device()
        
        # 将逻辑GPU ID转换为物理GPU ID
        physical_device = self.logical_to_physical.get(device, device)
        
        if physical_device not in self.devices:
            return 0
        
        with self.lock:
            total_pool_size = self.memory_pools[physical_device].numel()
            free_memory = self.get_free_memory(device)
            return total_pool_size - free_memory
    
    def print_memory_stats(self):
        """打印所有设备的内存统计信息"""
        for device in self.devices:
            total = self.memory_pools[device].numel()
            allocated = self.get_allocated_memory(device)
            free = self.get_free_memory(device)
            logical_id = self.physical_to_logical[device]
            
            print(f"物理设备 {device} (逻辑设备 {logical_id}) 内存统计:")
            print(f"  总池大小: {total/1024**3:.2f} GB")
            print(f"  已分配: {allocated/1024**3:.2f} GB ({allocated/total*100:.1f}%)")
            print(f"  可用: {free/1024**3:.2f} GB ({free/total*100:.1f}%)")
            print(f"  块数: {len(self.memory_blocks[device])}")
    
    def release(self):
        """释放所有预分配的内存"""
        # 释放所有预分配的tensor
        self._reserved_tensors.clear()
        _global_reserved_tensors.clear()  # 清理全局引用
        
        for device in self.devices:
            if device in self.memory_pools:
                del self.memory_pools[device]
        
        # 清理其他数据结构
        self.memory_blocks = {}
        self.allocation_map = {}
        
        # 强制垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
        
        if self.enable_logging:
            self.logger.info("GPU内存池已释放")
    
    def reserve_memory(self, device=None):
        """
        简化的方法：保留内存但不进行复杂的池管理
        这是一个更简单的方法，仅预分配内存以防止碎片化
        """
        if device is None:
            device = torch.cuda.current_device()
        
        # 将逻辑GPU ID转换为物理GPU ID
        physical_device = self.logical_to_physical.get(device, device)
        
        # 已经有预分配的内存池，不需要额外操作
        if physical_device in self.memory_pools:
            return
    
    @staticmethod
    def get_device_free_memory(device: int = None) -> int:
        """
        使用nvidia-smi命令获取设备上的空闲显存
        
        Args:
            device: GPU设备ID，None表示当前设备
            
        Returns:
            int: 空闲显存大小(字节)
        """
        if device is None:
            device = torch.cuda.current_device()
        
        try:
            # 使用nvidia-smi命令获取显存信息
            cmd = f"nvidia-smi --query-gpu=memory.free --format=csv,nounits -i {device}"
            result = subprocess.check_output(cmd, shell=True).decode('utf-8')
            
            # 解析输出获取空闲显存（MB）
            free_memory_mb = int(result.strip().split('\n')[1])
            
            # 转换为字节
            free_memory_bytes = free_memory_mb * 1024 * 1024
            
            return free_memory_bytes
            
        except Exception as e:
            print(f"获取GPU {device}空闲显存失败: {e}")
            # 如果nvidia-smi失败，回退到PyTorch的API，只能获取当前进程内部的分配情况
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            reserved_memory = torch.cuda.memory_reserved(device)
            return total_memory - allocated_memory - reserved_memory
    
    def _init_device_pool(self, device: int):
        """为特定设备初始化内存池"""
        # 使用物理GPU ID
        logical_device = self.physical_to_logical[device]
        torch.cuda.set_device(logical_device)
        torch.cuda.empty_cache()
        
        # 获取设备空闲显存
        free_memory = self.get_device_free_memory(device)
        allocate_memory = int(free_memory * self.allocation_ratio)
        
        if self.enable_logging:
            self.logger.info(f"物理设备 {device} (逻辑设备 {logical_device}): "
                           f"空闲显存 {free_memory/1024**3:.2f}GB, "
                           f"分配 {allocate_memory/1024**3:.2f}GB")
        
        # 预先分配大块内存
        try:
            # 使用整数类型减少内存消耗
            memory_pool = torch.empty(allocate_memory, dtype=torch.uint8, device=f"cuda:{logical_device}")
            self.memory_pools[device] = memory_pool
            self._reserved_tensors[device] = memory_pool  # 保持引用
            
            # 初始化为单一最大块
            self.memory_blocks[device] = [(0, allocate_memory)]  # (开始地址, 大小)
            self.allocation_map[device] = {}  # 地址 -> (大小, 引用)
            
            if self.enable_logging:
                self.logger.info(f"设备 {device}: 内存池创建成功")
                
        except RuntimeError as e:
            if self.enable_logging:
                self.logger.error(f"设备 {device}: 内存池创建失败: {e}")
            # 尝试分配较少的内存
            reduced_memory = int(allocate_memory * 0.9)
            if self.enable_logging:
                self.logger.info(f"设备 {device}: 尝试分配 {reduced_memory/1024**3:.2f}GB")
            memory_pool = torch.empty(reduced_memory, dtype=torch.uint8, device=f"cuda:{logical_device}")
            self.memory_pools[device] = memory_pool
            self._reserved_tensors[device] = memory_pool  # 保持引用
            self.memory_blocks[device] = [(0, reduced_memory)]
            self.allocation_map[device] = {}
    
    @staticmethod
    def simple_init(device_ids=None, mem_fraction=1.0):
        """
        简单初始化函数，一次性预留所有空闲显存，但允许PyTorch在训练中使用
        通过在缓存中预分配内存，然后释放引用，让PyTorch可以重用这部分内存
        
        Args:
            device_ids: GPU设备ID列表，None表示使用所有可用设备
            mem_fraction: 要使用的空闲显存比例(0.0-1.0)
        """
        # 获取CUDA_VISIBLE_DEVICES环境变量
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible_devices:
            try:
                # 解析环境变量中的GPU ID
                physical_gpus = [int(id.strip()) for id in visible_devices.split(',')]
                # 创建物理GPU ID到逻辑GPU ID的映射
                physical_to_logical = {physical: logical for logical, physical in enumerate(physical_gpus)}
                logical_to_physical = {logical: physical for physical, logical in physical_to_logical.items()}
            except:
                physical_gpus = list(range(torch.cuda.device_count()))
                physical_to_logical = {i: i for i in physical_gpus}
                logical_to_physical = {i: i for i in physical_gpus}
        else:
            physical_gpus = list(range(torch.cuda.device_count()))
            physical_to_logical = {i: i for i in physical_gpus}
            logical_to_physical = {i: i for i in physical_gpus}
        
        if device_ids is None:
            device_ids = physical_gpus
        
        print(f"物理GPU设备: {device_ids}")
        print(f"GPU ID映射: {physical_to_logical}")
        
        # 清空缓存
        torch.cuda.empty_cache()
        
        # 配置PyTorch的内存分配器
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
            'max_split_size_mb:512,'
            'garbage_collection_threshold:0.8'
        )
        
        # 为每个设备预留内存
        for physical_device in device_ids:
            logical_device = physical_to_logical[physical_device]
            
            # 获取设备空闲显存
            free_memory = GPUMemoryPool.get_device_free_memory(physical_device)
            memory_to_reserve = int(free_memory * mem_fraction)
            
            print(f"物理设备 {physical_device} (逻辑设备 {logical_device})："
                  f"空闲显存 {free_memory/1024**3:.2f}GB，"
                  f"预留 {memory_to_reserve/1024**3:.2f}GB")
            
            # 尝试预留内存：创建一个大的tensor，然后删除引用，内存会保留在CUDA缓存中
            try:
                # 分配多个中等大小的tensor，而不是一个大tensor
                chunk_size = 1024 * 1024 * 1024  # 1GB 块
                num_chunks = memory_to_reserve // chunk_size
                
                print(f"设备 {physical_device}: 分配 {num_chunks} 个 1GB 块")
                
                # 逐个分配并释放引用
                for i in range(num_chunks):
                    # 使用float32类型以减少内存占用
                    x = torch.zeros(chunk_size // 4, dtype=torch.float32, device=f'cuda:{logical_device}')
                    # 确保tensor被分配到GPU上
                    x.zero_()
                    # 释放引用，但内存会保留在CUDA缓存中
                    del x
                
                # 处理剩余部分
                remainder = memory_to_reserve % chunk_size
                if remainder > 0:
                    x = torch.zeros(remainder // 4, dtype=torch.float32, device=f'cuda:{logical_device}')
                    x.zero_()
                    del x
                
                print(f"设备 {physical_device}: 内存预留成功")
                
            except RuntimeError as e:
                print(f"设备 {physical_device} 内存预留失败: {e}")
                # 尝试分配更少的内存
                try:
                    reduced_memory = int(memory_to_reserve * 0.8)
                    print(f"尝试分配更少的内存: {reduced_memory/1024**3:.2f}GB")
                    
                    # 重试预留
                    chunk_size = 512 * 1024 * 1024  # 512MB 块
                    num_chunks = reduced_memory // chunk_size
                    
                    for i in range(num_chunks):
                        x = torch.zeros(chunk_size // 4, dtype=torch.float32, device=f'cuda:{logical_device}')
                        x.zero_()
                        del x
                    
                    # 处理剩余部分
                    remainder = reduced_memory % chunk_size
                    if remainder > 0:
                        x = torch.zeros(remainder // 4, dtype=torch.float32, device=f'cuda:{logical_device}')
                        x.zero_()
                        del x
                    
                except Exception as e2:
                    print(f"减少后的内存分配仍然失败: {e2}")
        
        # 打印当前状态
        for physical_device in device_ids:
            logical_device = physical_to_logical[physical_device]
            reserved = torch.cuda.memory_reserved(logical_device) / 1024**3
            allocated = torch.cuda.memory_allocated(logical_device) / 1024**3
            free = GPUMemoryPool.get_device_free_memory(physical_device) / 1024**3
            print(f"物理设备 {physical_device} (逻辑设备 {logical_device}) 状态: "
                  f"保留 {reserved:.2f}GB, 分配 {allocated:.2f}GB, 空闲 {free:.2f}GB")
            
            # 测试分配后的内存可用性
            try:
                test_tensor = torch.ones((1000, 1000), device=f'cuda:{logical_device}')
                del test_tensor
                print(f"设备 {physical_device}: 测试内存分配成功，内存缓存可用")
            except RuntimeError as e:
                print(f"设备 {physical_device}: 测试内存分配失败: {e}")
        
        return True

if __name__ == "__main__":
    # 测试GPU ID映射
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f"CUDA_VISIBLE_DEVICES: {visible_devices}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备数量: {torch.cuda.device_count()}")
    print(f"空闲显存: {GPUMemoryPool.get_device_free_memory()/(1024*1024*1024):.2f}GB")