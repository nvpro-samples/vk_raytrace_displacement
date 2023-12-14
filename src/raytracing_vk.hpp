/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file provides RAII-inspired C++ objects to facilitate creating and using
// a VkAccelerationStructureKHR object. This better demonstrates object
// dependencies and order of operations with more compile-time validation.
// Once a rt::BuiltAccelerationStructure is created it should be valid and
// usable in a raytracing pipeline. To do so, create...
//
// - rt::Context
// - rt::AccelerationStructureInput (optional utility container)
//   - E.g. using: rt::createBlasInput() for triangle geometry
//   - E.g. using: rt::createTlasInput() for BLAS instances
// - rt::AccelerationStructureSizes
// - rt::AccelerationStructure
// - rt::BuiltAccelerationStructure
//
// Finally, rt::BuiltAccelerationStructure provides a .object() and .address().

#pragma once

#include "nvvk/resourceallocator_vk.hpp"
#include <vulkan/vulkan_core.h>
#include <span>
#include <memory>
#include <functional>
#include <optional>
#include <nvvk/memallocator_vk.hpp>

namespace rt {

// Common container for the vulkan device and allocator. Many objects keep a
// reference to this to use during destruction, so it must outlive all other
// objects.
struct Context
{
  VkDevice                      device = VK_NULL_HANDLE;
  nvvk::ResourceAllocator*      allocator;
  const VkAllocationCallbacks*  allocationCallbacks = nullptr;
  std::function<void(VkResult)> resultCallback;
};

// Optional utility object to group source data for building and updating
// acceleration structures.
struct AccelerationStructureInput
{
  VkAccelerationStructureTypeKHR                        type;
  VkBuildAccelerationStructureFlagsKHR                  flags;
  std::vector<VkAccelerationStructureGeometryKHR>       geometries;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR> rangeInfos;
};

// VkAccelerationStructureBuildSizesInfoKHR wrapper, a dependency of the main
// AccelerationStructure.
class AccelerationStructureSizes
{
public:
  AccelerationStructureSizes(const Context& context, const AccelerationStructureInput& input)
      : AccelerationStructureSizes(context, input.type, input.flags, input.geometries, input.rangeInfos)
  {
  }
  AccelerationStructureSizes(const Context&                                            context,
                             VkAccelerationStructureTypeKHR                            type,
                             VkBuildAccelerationStructureFlagsKHR                      flags,
                             std::span<const VkAccelerationStructureGeometryKHR>       geometries,
                             std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos)
      : m_sizeInfo{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR}
  {
    assert(geometries.size() == rangeInfos.size());
    VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{
        .sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type          = type,
        .flags         = flags,
        .geometryCount = static_cast<uint32_t>(geometries.size()),
        .pGeometries   = geometries.data(),
    };
    std::vector<uint32_t> primitiveCounts(rangeInfos.size());
    std::transform(rangeInfos.begin(), rangeInfos.end(), primitiveCounts.begin(),
                   [](const VkAccelerationStructureBuildRangeInfoKHR& rangeInfo) { return rangeInfo.primitiveCount; });
    vkGetAccelerationStructureBuildSizesKHR(context.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildGeometryInfo, primitiveCounts.data(), &m_sizeInfo);
  }
  const VkAccelerationStructureBuildSizesInfoKHR& operator*() const { return m_sizeInfo; }
  VkAccelerationStructureBuildSizesInfoKHR&       operator*() { return m_sizeInfo; }
  const VkAccelerationStructureBuildSizesInfoKHR* operator->() const { return &m_sizeInfo; }
  VkAccelerationStructureBuildSizesInfoKHR*       operator->() { return &m_sizeInfo; }

private:
  VkAccelerationStructureBuildSizesInfoKHR m_sizeInfo;
};

// RAII VkBuffer wrapper.
class Buffer
{
public:
  Buffer(const Buffer& other) = delete;
  Buffer(Buffer&& other)
      : m_context(std::move(other.m_context))
      , m_buffer(std::move(other.m_buffer))
      , m_address(std::move(other.m_address))
  {
    other.m_context = nullptr;
    other.m_buffer  = {};
    other.m_address = {};
  }
  Buffer(const Context& context, VkDeviceSize size, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags)
      : m_context(&context)
      , m_buffer(context.allocator->createBuffer(size, usageFlags, propertyFlags))
      , m_address(getAddress(context.device, m_buffer.buffer))
  {
    VkBufferDeviceAddressInfo bufferInfo{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = m_buffer.buffer,
    };
    m_address = vkGetBufferDeviceAddress(m_context->device, &bufferInfo);
  }
  template <class Range>
  Buffer(const Context& context, const Range& range, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, VkCommandBuffer cmd)
      : m_context(&context)
      , m_buffer(context.allocator->createBuffer(cmd, sizeof(*range.data()) * range.size(), range.data(), usageFlags, propertyFlags))
      , m_address(getAddress(context.device, m_buffer.buffer))
  {
  }
  ~Buffer()
  {
    if(m_context)
    {
      m_context->allocator->destroy(m_buffer);
    }
  }
  Buffer&                operator=(const Buffer& other) = delete;
  Buffer&                operator=(Buffer&& other)      = delete;
  const VkDeviceAddress& address() const { return m_address; }
  operator const VkBuffer&() { return m_buffer.buffer; }

private:
  static VkDeviceAddress getAddress(VkDevice device, VkBuffer buffer)
  {
    VkBufferDeviceAddressInfo bufferInfo{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = buffer,
    };
    return vkGetBufferDeviceAddress(device, &bufferInfo);
  }

  const Context*  m_context = nullptr;
  nvvk::Buffer    m_buffer;
  VkDeviceAddress m_address;
};

// Optional utility wrapper for creating and populating a Buffer of instances
// for a top level accelreation structure build.
class InstanceBuffer : public Buffer
{
public:
  InstanceBuffer(const Context& context, std::span<const VkAccelerationStructureInstanceKHR> instances, VkCommandBuffer cmd)
      : Buffer(context,
               instances,
               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
               cmd)
  {
    // Make sure Buffer()'s upload is complete and visible for the subsequent
    // acceleration structure build.
    VkMemoryBarrier barrier{
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0,
                         1, &barrier, 0, nullptr, 0, nullptr);
  }
};

// VkAccelerationStructureKHR wrapper including a Buffer that holds backing
// memory for the vulkan object itself and the built acceleration structure.
// This can be a top or bottom level acceleration structure depending on the
// 'type' passed to the constructor. To use the acceleration structure it must
// first be given to BuiltAccelerationStructure.
class BuiltAccelerationStructure;
class AccelerationStructure
{
public:
  AccelerationStructure()                                   = delete;
  AccelerationStructure(const AccelerationStructure& other) = delete;
  AccelerationStructure(AccelerationStructure&& other)
      : m_context(std::move(other.m_context))
      , m_buffer(std::move(other.m_buffer))
      , m_type(std::move(other.m_type))
      , m_size(std::move(other.m_size))
      , m_accelerationStructure(std::move(other.m_accelerationStructure))
      , m_address(std::move(other.m_address))
  {
    other.m_context               = nullptr;
    other.m_type                  = {};
    other.m_size                  = {};
    other.m_accelerationStructure = VK_NULL_HANDLE;
    other.m_address               = {};
  }
  AccelerationStructure(const Context&                                  context,
                        VkAccelerationStructureTypeKHR                  type,
                        const VkAccelerationStructureBuildSizesInfoKHR& size,
                        VkAccelerationStructureCreateFlagsKHR           flags)
      : m_context(&context)
      , m_type(type)
      , m_size(size)
      , m_buffer(context,
                 m_size.accelerationStructureSize,
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
      , m_accelerationStructure(VK_NULL_HANDLE)
  {
    VkAccelerationStructureCreateInfoKHR createInfo{
        .sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .createFlags   = flags,
        .buffer        = m_buffer,
        .offset        = 0,
        .size          = m_size.accelerationStructureSize,
        .type          = m_type,
        .deviceAddress = 0,
    };
    context.resultCallback(vkCreateAccelerationStructureKHR(context.device, &createInfo, context.allocationCallbacks,
                                                            &m_accelerationStructure));

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
        .sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = m_accelerationStructure,
    };
    m_address = vkGetAccelerationStructureDeviceAddressKHR(context.device, &addressInfo);
  }
  ~AccelerationStructure()
  {
    if(m_context)
    {
      vkDestroyAccelerationStructureKHR(m_context->device, m_accelerationStructure, m_context->allocationCallbacks);
    }
  }
  AccelerationStructure&                          operator=(const AccelerationStructure& other) = delete;
  AccelerationStructure&                          operator=(AccelerationStructure&& other)      = delete;
  const VkAccelerationStructureTypeKHR&           type() const { return m_type; }
  const VkAccelerationStructureBuildSizesInfoKHR& sizes() { return m_size; }

private:
  // Use the C++ type system to hide access to the object until it is built with
  // BuiltAccelerationStructure. This adds a little compile-time state checking.
  friend class BuiltAccelerationStructure;
  const VkAccelerationStructureKHR& object() const { return m_accelerationStructure; }
  const VkDeviceAddress&            address() const { return m_address; }

  const Context*                           m_context;
  VkAccelerationStructureTypeKHR           m_type;
  VkAccelerationStructureBuildSizesInfoKHR m_size;
  Buffer                                   m_buffer;
  VkAccelerationStructureKHR               m_accelerationStructure;
  VkDeviceAddress                          m_address;
};

// A growable Buffer with flags suitable for acceleration structure builds.
class ScratchBuffer
{
public:
  ScratchBuffer(const Context& context)
      : m_context(&context)
  {
  }
  ScratchBuffer(const Context& context, VkDeviceSize size)
      : m_context(&context)
  {
    resize(size);
  }
  const Buffer& buffer(VkDeviceSize size)
  {
    if(m_size < size)
      resize(size);
    return *m_buffer;
  }

private:
  void resize(VkDeviceSize size)
  {
    m_size = size;
    m_buffer.emplace(*m_context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }

  const Context*        m_context;
  VkDeviceSize          m_size = 0;
  std::optional<Buffer> m_buffer;
};

// An AccelerationStructure that is guaranteed to have had
// vkCmdBuildAccelerationStructuresKHR called on it. An AccelerationStructure
// must be std::move()d into the constructor, along with the inputs for the
// build. An update() call is also provided, which uses
// VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR and writes the updated data
// into the same object. Note that it is up to the caller to make sure to submit
// the command buffer, filled as a side effect of the constructor, before
// submitting operations that use the built acceleration structure.
class BuiltAccelerationStructure
{
public:
  BuiltAccelerationStructure(const Context&                    context,
                             AccelerationStructure&&           accelerationStructure,
                             const AccelerationStructureInput& input,
                             ScratchBuffer&                    scratchBuffer,
                             VkCommandBuffer                   cmd)
      : BuiltAccelerationStructure(context, std::move(accelerationStructure), input.flags, input.geometries, input.rangeInfos, scratchBuffer, cmd)
  {
  }
  BuiltAccelerationStructure(const Context&                                            context,
                             AccelerationStructure&&                                   accelerationStructure,
                             VkBuildAccelerationStructureFlagsKHR                      flags,
                             std::span<const VkAccelerationStructureGeometryKHR>       geometries,
                             std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos,
                             ScratchBuffer&                                            scratchBuffer,
                             VkCommandBuffer                                           cmd)
      : m_accelerationStructure(std::move(accelerationStructure))
  {
    build(context, flags, geometries, rangeInfos, false, scratchBuffer, cmd);
  }

  // Build again from scratch calling vkCmdBuildAccelerationStructuresKHR()
  // without VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR.
  void rebuild(const Context& context, const AccelerationStructureInput& input, ScratchBuffer& scratchBuffer, VkCommandBuffer cmd)
  {
    rebuild(context, input.flags, input.geometries, input.rangeInfos, scratchBuffer, cmd);
  }

  // Build again from scratch calling vkCmdBuildAccelerationStructuresKHR()
  // without VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR.
  void rebuild(const Context&                                            context,
               VkBuildAccelerationStructureFlagsKHR                      flags,
               std::span<const VkAccelerationStructureGeometryKHR>       geometries,
               std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos,
               ScratchBuffer&                                            scratchBuffer,
               VkCommandBuffer                                           cmd)
  {
    build(context, flags, geometries, rangeInfos, false, scratchBuffer, cmd);
  }

  // Update the built structure calling vkCmdBuildAccelerationStructuresKHR()
  // with VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR.
  void update(const Context& context, const AccelerationStructureInput& input, ScratchBuffer& scratchBuffer, VkCommandBuffer cmd)
  {
    update(context, input.flags, input.geometries, input.rangeInfos, scratchBuffer, cmd);
  }

  // Update the built structure calling vkCmdBuildAccelerationStructuresKHR()
  // with VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR.
  void update(const Context&                                            context,
              VkBuildAccelerationStructureFlagsKHR                      flags,
              std::span<const VkAccelerationStructureGeometryKHR>       geometries,
              std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos,
              ScratchBuffer&                                            scratchBuffer,
              VkCommandBuffer                                           cmd)
  {
    build(context, flags, geometries, rangeInfos, true, scratchBuffer, cmd);
  }

  operator const VkAccelerationStructureKHR&() const { return m_accelerationStructure.object(); }
  const VkAccelerationStructureKHR& object() const { return m_accelerationStructure.object(); }
  const VkDeviceAddress&            address() const { return m_accelerationStructure.address(); }

private:
  void build(const Context&                                            context,
             VkBuildAccelerationStructureFlagsKHR                      flags,
             std::span<const VkAccelerationStructureGeometryKHR>       geometries,
             std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos,
             bool                                                      update,
             ScratchBuffer&                                            scratchBuffer,
             VkCommandBuffer                                           cmd)
  {
    assert(geometries.size() == rangeInfos.size());
    assert(!update || !!(flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR));
    VkBuildAccelerationStructureModeKHR mode =
        update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    auto&                                       sizes       = m_accelerationStructure.sizes();
    VkDeviceSize                                scratchSize = update ? sizes.updateScratchSize : sizes.buildScratchSize;
    VkDeviceAddress                             scratchAddress = scratchBuffer.buffer(scratchSize).address();
    VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{
        .sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type                     = m_accelerationStructure.type(),
        .flags                    = flags,
        .mode                     = mode,
        .srcAccelerationStructure = update ? m_accelerationStructure.object() : VK_NULL_HANDLE,
        .dstAccelerationStructure = m_accelerationStructure.object(),
        .geometryCount            = static_cast<uint32_t>(geometries.size()),
        .pGeometries              = geometries.data(),
        .scratchData              = {.deviceAddress = scratchAddress},
    };
    auto rangeInfosPtr = rangeInfos.data();
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildGeometryInfo, &rangeInfosPtr);

    // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
    // is finished before starting the next one.
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  }

  AccelerationStructure m_accelerationStructure;
};

// Optional utility call to fill a AccelerationStructureInput with instances for
// a top level acceleration structure build and update.
AccelerationStructureInput createTlasInput(uint32_t instanceCount, VkDeviceAddress instanceBufferAddress, VkBuildAccelerationStructureFlagsKHR flags)
{
  return AccelerationStructureInput{
      .type  = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      .flags = flags,
      .geometries{
          VkAccelerationStructureGeometryKHR{
              .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
              .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
              .geometry =
                  VkAccelerationStructureGeometryDataKHR{
                      .instances =
                          VkAccelerationStructureGeometryInstancesDataKHR{
                              .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                              .data  = {instanceBufferAddress},
                          },
                  },
          },
      },
      .rangeInfos{
          VkAccelerationStructureBuildRangeInfoKHR{
              .primitiveCount = instanceCount,
          },
      },
  };
}

// Essentially just VkAccelerationStructureGeometryTrianglesDataKHR with some
// defaults and geometryFlags.
struct SimpleGeometryInput
{
  uint32_t           triangleCount;
  uint32_t           maxVertex;
  VkDeviceAddress    indexAddress;
  VkDeviceAddress    vertexAddress;
  VkDeviceSize       vertexStride  = sizeof(float) * 3;
  VkFormat           vertexFormat  = VK_FORMAT_R32G32B32_SFLOAT;
  VkIndexType        indexType     = VK_INDEX_TYPE_UINT32;
  VkGeometryFlagsKHR geometryFlags = VK_GEOMETRY_OPAQUE_BIT_KHR | VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
};

// Optional utility call to fill a AccelerationStructureInput with triangle
// geometry for a bottom level acceleration structure build and update.
AccelerationStructureInput createBlasInput(std::span<const SimpleGeometryInput> simpleInputs,
                                           VkBuildAccelerationStructureFlagsKHR accelerationStructureFlags)
{
  AccelerationStructureInput result{
      .type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      .flags = accelerationStructureFlags,
  };
  for(const auto& simpleInput : simpleInputs)
  {
    result.geometries.emplace_back(VkAccelerationStructureGeometryKHR{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry =
            VkAccelerationStructureGeometryDataKHR{
                .triangles =
                    VkAccelerationStructureGeometryTrianglesDataKHR{
                        .sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                        .vertexFormat  = simpleInput.vertexFormat,
                        .vertexData    = {simpleInput.vertexAddress},
                        .vertexStride  = simpleInput.vertexStride,
                        .maxVertex     = simpleInput.maxVertex,
                        .indexType     = simpleInput.indexType,
                        .indexData     = {simpleInput.indexAddress},
                        .transformData = {0},
                    },
            },
        .flags = simpleInput.geometryFlags,
    });
    result.rangeInfos.emplace_back(VkAccelerationStructureBuildRangeInfoKHR{
        .primitiveCount = simpleInput.triangleCount,
    });
  }
  return result;
}

}  // namespace rt
