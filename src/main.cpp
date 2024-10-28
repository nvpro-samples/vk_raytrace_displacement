/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
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

//////////////////////////////////////////////////////////////////////////
/*

  This sample demonstrates the use of the `heightmap_rtx` library to raytrace
  dynamically displaced geometry. The key areas are:
  - createHrtxPipeline()
  - createHrtxMap()
  - The HrtxMap usage in createBottomLevelAS()
  - The additional submit(1, &m_cmdHrtxUpdate) in onRender() for animation

  This sample's code is branched from the code at:
  https://github.com/NVIDIAGameWorks/Displacement-MicroMap-Toolkit/tree/main/mini_samples/dmm_displacement

*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <cstddef>
#include <memory>
#include <chrono>
#include <random>
#include <glm/vec4.hpp>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#include "nvmath/nvmath_types.h"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/memallocator_vma_vk.hpp"
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/primitives.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"

#include "shaders/dh_bindings.h"
#include "shaders/device_host.h"
#include "shaders/animate_heightmap.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "third_party/imgui/backends/imgui_impl_vulkan.h"

#include "heightmap_rtx.h"
#include "raytracing_vk.hpp"

// Pre-compiled SPIR-V, see CMakeLists.txt
#include "generated_spirv/animate_heightmap.comp.h"
#include "generated_spirv/pathtrace.rchit.h"
#include "generated_spirv/pathtrace.rgen.h"
#include "generated_spirv/pathtrace.rmiss.h"

#define HEIGHTMAP_RESOLUTION 256

// Move-only VkShaderModule constructed from SPIR-V data
class ShaderModule
{
public:
  ShaderModule() = default;
  template <size_t N>
  ShaderModule(VkDevice device, const uint32_t (&spirv)[N])
      : m_device(device)
      , m_module(nvvk::createShaderModule(device, spirv, N * sizeof(uint32_t)))
  {
  }
  ~ShaderModule() { destroy(); }
  ShaderModule(const ShaderModule& other) = delete;
  ShaderModule(ShaderModule&& other)
      : m_device(other.m_device)
      , m_module(other.m_module)
  {
    other.m_module = VK_NULL_HANDLE;
  }
  ShaderModule& operator=(const ShaderModule& other) = delete;
  ShaderModule& operator=(ShaderModule&& other)
  {
    destroy();
    m_module = VK_NULL_HANDLE;
    std::swap(m_module, other.m_module);
    m_device = other.m_device;
    return *this;
  }
  operator VkShaderModule() const { return m_module; }

private:
  void destroy()
  {
    if(m_module != VK_NULL_HANDLE)
      vkDestroyShaderModule(m_device, m_module, nullptr);
  }
  VkDevice       m_device = VK_NULL_HANDLE;
  VkShaderModule m_module = VK_NULL_HANDLE;
};

// Container to run a compute shader with only a single instance of bindings.
template <class PushConstants>
struct SingleComputePipeline
{
  VkDescriptorSet       descriptorSet{VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
  VkDescriptorPool      descriptorPool{VK_NULL_HANDLE};
  VkPipeline            pipeline{VK_NULL_HANDLE};
  VkPipelineLayout      pipelineLayout{VK_NULL_HANDLE};
  ShaderModule          shaderModule;

  // Slightly ugly callback for declaring and writing shader bindings, allowing
  // this class to be more generic. A better way would be to split up the object
  // and pass initialized objects to the constructor.
  struct BindingsCB
  {
    std::function<void(nvvk::DescriptorSetBindings&)>                                               declare;
    std::function<std::vector<VkWriteDescriptorSet>(nvvk::DescriptorSetBindings&, VkDescriptorSet)> create;
  };

  void create(VkDevice device, const BindingsCB& bindingsCB, ShaderModule&& module)
  {
    shaderModule = std::move(module);

    nvvk::DescriptorSetBindings bindings;
    bindingsCB.declare(bindings);

    descriptorSetLayout = bindings.createLayout(device);
    descriptorPool      = bindings.createPool(device);
    descriptorSet       = nvvk::allocateDescriptorSet(device, descriptorPool, descriptorSetLayout);

    std::vector<VkWriteDescriptorSet> bindingsDescWrites = bindingsCB.create(bindings, descriptorSet);
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(bindingsDescWrites.size()), bindingsDescWrites.data(), 0, nullptr);

    VkPushConstantRange pushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo pipelineLayoutCreate{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRange,
    };
    vkCreatePipelineLayout(device, &pipelineLayoutCreate, nullptr, &pipelineLayout);

    VkPipelineShaderStageCreateInfo shaderStageCreate{
        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shaderModule,
        .pName  = "main",
    };

    VkComputePipelineCreateInfo pipelineCreate{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = shaderStageCreate,
        .layout = pipelineLayout,
    };

    vkCreateComputePipelines(device, {}, 1, &pipelineCreate, nullptr, &pipeline);
  }

  void dispatch(VkCommandBuffer cmd, const PushConstants pushConstants, uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1)
  {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushConstants);
    vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);
  }

  void destroy(VkDevice device)
  {
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    shaderModule = ShaderModule();
  }
};

struct AnimatedHeightmap
{
  void create(nvvkhl::AllocVma& alloc, nvvk::DebugUtil& dutil, uint32_t resolution)
  {
    m_resolution = resolution;
    createHeightmaps(alloc, dutil);

    m_animatePipeline.create(
        alloc.getDevice(),
        {[](nvvk::DescriptorSetBindings& bindings) {
           bindings.addBinding(BINDING_ANIM_IMAGE_A_HEIGHT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
           bindings.addBinding(BINDING_ANIM_IMAGE_B_HEIGHT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
           bindings.addBinding(BINDING_ANIM_IMAGE_A_VELOCITY, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
           bindings.addBinding(BINDING_ANIM_IMAGE_B_VELOCITY, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
         },
         [this](nvvk::DescriptorSetBindings& bindings, VkDescriptorSet descriptorSet) {
           return std::vector<VkWriteDescriptorSet>{
               bindings.makeWrite(descriptorSet, BINDING_ANIM_IMAGE_A_HEIGHT, &m_heightmapA.descriptor),
               bindings.makeWrite(descriptorSet, BINDING_ANIM_IMAGE_B_HEIGHT, &m_heightmapB.descriptor),
               bindings.makeWrite(descriptorSet, BINDING_ANIM_IMAGE_A_VELOCITY, &m_velocityA.descriptor),
               bindings.makeWrite(descriptorSet, BINDING_ANIM_IMAGE_B_VELOCITY, &m_velocityB.descriptor),
           };
         }},
        ShaderModule(alloc.getDevice(), animate_heightmap_comp));
  }
  void destroy(nvvkhl::AllocVma& alloc)
  {
    destroyHeightmaps(alloc);
    m_animatePipeline.destroy(alloc.getDevice());
  }

  void clear(VkCommandBuffer cmd)
  {
    VkClearColorValue       heightValue{.float32 = {0.5f}};
    VkClearColorValue       velocityValue{.float32 = {0.0f}};
    VkImageSubresourceRange range{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};

    // Perform the initial transition from VK_IMAGE_LAYOUT_UNDEFINED
    imageBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL);
    m_currentIsA = !m_currentIsA;
    imageBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL);

    vkCmdClearColorImage(cmd, height().image, height().descriptor.imageLayout, &heightValue, 1, &range);
    vkCmdClearColorImage(cmd, velocity().image, velocity().descriptor.imageLayout, &velocityValue, 1, &range);
    imageBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL);
    m_currentIsA = !m_currentIsA;
  }

  void animate(VkCommandBuffer cmd)
  {
    shaders::AnimatePushConstants pushConstants{
        .mouse      = m_mouse * glm::vec2(m_resolution),
        .writeToA   = m_currentIsA ? 0U : 1U,
        .resolution = int(m_resolution),
        .deltaTime  = 1.0f,
    };

    // Add some raindrops if the user doesn't draw for a few seconds
    const double timeout      = 5.0;
    const double dropDelay    = 0.5;
    const double dropDuration = 0.05;
    auto         now          = std::chrono::system_clock::now();
    static auto  lastDraw     = std::chrono::system_clock::time_point{};
    if(m_mouse.x >= 0.0f)
    {
      lastDraw = now;
    }
    if(std::chrono::duration<double>(now - lastDraw).count() > timeout)
    {
      static std::random_device                    rd;
      static std::mt19937                          mt(rd());
      static std::uniform_real_distribution<float> dist(0.0, double(m_resolution));
      static auto                                  lastDrop = now;
      static glm::vec2                             dropPos  = {};
      if(std::chrono::duration<double>(now - lastDrop).count() > dropDelay)
      {
        lastDrop = now;
        dropPos  = {dist(mt), dist(mt)};
      }
      if(std::chrono::duration<double>(now - lastDrop).count() < dropDuration)
      {
        pushConstants.mouse = dropPos;
      }
    }

    assert(m_resolution % ANIMATION_WORKGROUP_SIZE == 0);
    m_animatePipeline.dispatch(cmd, pushConstants, m_resolution / ANIMATION_WORKGROUP_SIZE, m_resolution / ANIMATION_WORKGROUP_SIZE);

    imageBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL);

    m_currentIsA = !m_currentIsA;
  }

  const nvvk::Texture& height() const { return m_currentIsA ? m_heightmapB : m_heightmapA; }
  const nvvk::Texture& velocity() const { return m_currentIsA ? m_velocityB : m_velocityA; }
  void                 setMouse(const glm::vec2& position) { m_mouse = position; }

private:
  VkImageLayout& imageLayouts() { return m_currentIsA ? m_currentImageLayoutsB : m_currentImageLayoutsA; }
  void           imageBarrier(VkCommandBuffer cmd, VkImageLayout newLayout)
  {
    VkImageMemoryBarrier barriers[] = {
        nvvk::makeImageMemoryBarrier(height().image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, imageLayouts(), newLayout),
        nvvk::makeImageMemoryBarrier(velocity().image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, imageLayouts(), newLayout),
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, std::size(barriers), barriers);
    imageLayouts() = newLayout;
  }

  void createHeightmaps(nvvkhl::AllocVma& alloc, nvvk::DebugUtil& dutil)
  {
    VkExtent2D          extent      = {m_resolution, m_resolution};
    VkFormat            format      = VK_FORMAT_R32_SFLOAT;
    VkImageUsageFlags   flags       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    VkImageCreateInfo   imageInfo   = nvvk::makeImage2DCreateInfo(extent, format, flags);
    nvvk::Image         imageA      = alloc.createImage(imageInfo);
    nvvk::Image         imageB      = alloc.createImage(imageInfo);
    nvvk::Image         imageC      = alloc.createImage(imageInfo);
    nvvk::Image         imageD      = alloc.createImage(imageInfo);
    VkSamplerCreateInfo samplerInfo = {.sType      = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                       .magFilter  = VK_FILTER_LINEAR,
                                       .minFilter  = VK_FILTER_LINEAR,
                                       .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                       .maxLod     = FLT_MAX};

    m_heightmapA = alloc.createTexture(imageA, nvvk::makeImage2DViewCreateInfo(imageA.image, format), samplerInfo);
    m_heightmapB = alloc.createTexture(imageB, nvvk::makeImage2DViewCreateInfo(imageB.image, format), samplerInfo);
    m_velocityA  = alloc.createTexture(imageC, nvvk::makeImage2DViewCreateInfo(imageC.image, format), samplerInfo);
    m_velocityB  = alloc.createTexture(imageD, nvvk::makeImage2DViewCreateInfo(imageD.image, format), samplerInfo);
    dutil.setObjectName(m_heightmapA.descriptor.imageView, "HeightmapA");
    dutil.setObjectName(m_heightmapB.descriptor.imageView, "HeightmapB");
    dutil.setObjectName(m_velocityA.descriptor.imageView, "VelocityA");
    dutil.setObjectName(m_velocityB.descriptor.imageView, "VelocityB");

    // Image layouts can change over time. Despite this, nvvk::Texture holds a
    // layout in a persistent descriptor which isn't always kept up to date.
    // nvvk::ResourceAllocator defaults to
    // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL but since we know we'll
    // transition to VK_IMAGE_LAYOUT_GENERAL, this is set before creating the
    // pipeline descriptor set layout.
    m_heightmapA.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    m_heightmapB.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    m_velocityA.descriptor.imageLayout  = VK_IMAGE_LAYOUT_GENERAL;
    m_velocityB.descriptor.imageLayout  = VK_IMAGE_LAYOUT_GENERAL;
  }

  void destroyHeightmaps(nvvkhl::AllocVma& alloc)
  {
    alloc.destroy(m_heightmapA);
    alloc.destroy(m_heightmapB);
    alloc.destroy(m_velocityA);
    alloc.destroy(m_velocityB);
  }

  SingleComputePipeline<shaders::AnimatePushConstants> m_animatePipeline;

  uint32_t      m_resolution;
  nvvk::Texture m_heightmapA;
  nvvk::Texture m_heightmapB;
  nvvk::Texture m_velocityA;
  nvvk::Texture m_velocityB;
  bool          m_currentIsA = true;
  glm::vec2     m_mouse;
  VkImageLayout m_currentImageLayoutsA = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout m_currentImageLayoutsB = VK_IMAGE_LAYOUT_UNDEFINED;
};

class RaytracingSample : public nvvkhl::IAppElement
{
  struct Settings
  {
    float opacity{0.25F};
    float refractiveIndex{1.03F};
    float density{1.0F};
    float heightmapScale{0.2f};
    int   maxDepth{5};
    bool  enableAnimation{true};
    bool  enableDisplacement{true};
    int   subdivlevel{3};
  } m_settings;

  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };

public:
  RaytracingSample()           = default;
  ~RaytracingSample() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice         = app->getPhysicalDevice();
    allocator_info.device                 = app->getDevice();
    allocator_info.instance               = app->getInstance();
    allocator_info.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    m_dutil             = std::make_unique<nvvk::DebugUtil>(m_device);         // Debug utility
    m_alloc             = std::make_unique<nvvkhl::AllocVma>(allocator_info);  // Allocator
    m_staticCommandPool = std::make_unique<nvvk::CommandPool>(m_device, m_app->getQueue(0).queueIndex);
    m_rtContext         = {m_device, m_alloc.get(), nullptr, [](VkResult result) { NVVK_CHECK(result); }};
    m_rtScratchBuffer   = std::make_unique<rt::ScratchBuffer>(m_rtContext);

    m_rtSet.init(m_device);

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shader Binding Table (SBT)
    int32_t gct_queue_index = m_app->getQueue(0).familyIndex;
    m_sbt.setup(m_device, gct_queue_index, m_alloc.get(), m_rtProperties);

    // Create resources
    createScene();
    createVkBuffers();
    createHrtxPipeline();
    static_assert(HEIGHTMAP_RESOLUTION % ANIMATION_WORKGROUP_SIZE == 0, "currently, resolution must match compute workgroup size");
    m_heightmap.create(*m_alloc, *m_dutil, HEIGHTMAP_RESOLUTION);
    const VkDescriptorImageInfo& heightmapHeightDesc = m_heightmap.height().descriptor;  // same for both buffers A/B
    m_heightmapImguiDesc = ImGui_ImplVulkan_AddTexture(heightmapHeightDesc.sampler, heightmapHeightDesc.imageView,
                                                       heightmapHeightDesc.imageLayout);

    // Initialize the heightmap textures before referencing one in
    // createHrtxMap().
    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      m_heightmap.clear(cmd);
      m_app->submitAndWaitTempCmdBuffer(cmd);
    }

    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      createBottomLevelAS(cmd);
      createTopLevelAS(cmd);
      m_app->submitAndWaitTempCmdBuffer(cmd);
    }

    createRtxPipeline();
    createGbuffers(m_viewSize);
  }

  void onDetach() override { destroyResources(); }

  void onResize(uint32_t width, uint32_t height) override
  {
    createGbuffers({width, height});
    writeRtDesc();
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");
      ImGuiH::CameraWidget();

      using namespace ImGuiH;

      bool recreateAS = false;

      // #MICROMESH - begin
      ImGui::Text("Heightmap Displacement");
      PropertyEditor::begin();
      recreateAS |=
          PropertyEditor::entry("Enable", [&] { return ImGui::Checkbox("##ll", &m_settings.enableDisplacement); });
      PropertyEditor::entry("Animation", [&] { return ImGui::Checkbox("##ll", &m_settings.enableAnimation); });

      recreateAS |= PropertyEditor::entry("Subdivision Level",
                                          [&] { return ImGui::SliderInt("#1", &m_settings.subdivlevel, 0, 5); });
      recreateAS |= PropertyEditor::entry("Heightmap Scale", [&] {
        return ImGui::SliderFloat("#1", &m_settings.heightmapScale, 0.05F, 2.0F);
      });

      if(recreateAS)
      {
        vkDeviceWaitIdle(m_device);

        // HrtxMap objects need to be re-created when the input attributes
        // change
        destroyHrtxMaps();

        // Recreate the acceleration structure
        VkCommandBuffer initCmd = m_app->createTempCmdBuffer();
        createBottomLevelAS(initCmd);
        createTopLevelAS(initCmd);
        m_app->submitAndWaitTempCmdBuffer(initCmd);
        writeRtDesc();
      }
      // #MICROMESH - end

      PropertyEditor::end();
      ImGui::Text("Material");
      PropertyEditor::begin();
      PropertyEditor::entry("Opacity", [&] { return ImGui::SliderFloat("#1", &m_settings.opacity, 0.0F, 1.0F); });
      PropertyEditor::entry("Refractive Index",
                            [&] { return ImGui::SliderFloat("#1", &m_settings.refractiveIndex, 0.5F, 4.0F); });
      PropertyEditor::entry("Density", [&] { return ImGui::SliderFloat("#1", &m_settings.density, 0.0F, 5.0F); });
      PropertyEditor::end();
      ImGui::Separator();
      ImGui::Text("Sun Orientation");
      PropertyEditor::begin();
      glm::vec3 dir = m_skyParams.directionToLight;
      ImGuiH::azimuthElevationSliders(dir, false);
      m_skyParams.directionToLight = dir;
      PropertyEditor::end();
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffer->getDescriptorSet(), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }

    {  // Heightmap preview and mouse interaction
      ImGui::Begin("Heightmap");
      ImGui::Text("Animated heightmap. Click to draw.");

      ImVec2 windowPos       = ImGui::GetCursorScreenPos();
      ImVec2 windowSize      = ImGui::GetContentRegionAvail();
      ImVec2 previewSize     = {std::min(windowSize.x, windowSize.y), std::min(windowSize.x, windowSize.y)};
      ImVec2 marginTotal     = {windowSize.x - previewSize.x, windowSize.y - previewSize.y};
      ImVec2 heightmapOffset = {marginTotal.x / 2, marginTotal.y / 2};
      auto   mouseAbs        = ImGui::GetIO().MousePos;
      ImVec2 mouse     = {mouseAbs.x - heightmapOffset.x - windowPos.x, mouseAbs.y - heightmapOffset.y - windowPos.y};
      auto   mouseNorm = glm::vec2{mouse.x, mouse.y} / glm::vec2{previewSize.x, previewSize.y};

      // Update the heightmap mouse position when dragging the mouse in the
      // heightmap window. If not clicking, moving the mouse off-screen will
      // stop it affecting the animation
      if(ImGui::GetIO().MouseDown[0] && mouseNorm.x >= 0.0 && mouseNorm.x <= 1.0f && mouseNorm.y >= 0.0f && mouseNorm.y < 1.0f)
      {
        m_heightmap.setMouse(mouseNorm);
      }
      else
      {
        m_heightmap.setMouse(glm::vec2(-0.5f));
      }

      // Display the heightmap
      ImVec2 drawPos = ImGui::GetCursorPos();
      ImGui::SetCursorPos({heightmapOffset.x + drawPos.x, heightmapOffset.y + drawPos.y});
      ImGui::Image(m_heightmapImguiDesc, previewSize);

      ImGui::End();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    if(m_settings.enableAnimation)
    {
      // Step the heightmap animation before submitting m_cmdHrtxUpdate. Note
      // that animate() is called twice to get the double buffered results back
      // into the original. Ideally this would just be submitTempCmdBuffer() to
      // avoid a GPU stall in the render loop, but that call doesn't exist yet.
      VkCommandBuffer animCmd = m_app->createTempCmdBuffer();
      m_heightmap.animate(animCmd);
      m_heightmap.animate(animCmd);
      m_app->submitAndWaitTempCmdBuffer(animCmd);

      // Update the raytracing displacement from the heightmap. m_cmdHrtxUpdate
      // already includes an image barrier for compute shader writes.
      VkSubmitInfo submit = {
          .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
          .commandBufferCount = 1,
          .pCommandBuffers    = &m_cmdHrtxUpdate,
      };
      vkQueueSubmit(m_app->getQueue(0).queue, 1, &submit, VK_NULL_HANDLE);

      // Rebuild the BLAS for the heightmap. The TLAS then needs updating too,
      // but not the other static geometry. Note that the above animation and
      // heightmap_rtx commands are submitted on the same queue before before
      // onRender() returns and the temporary 'cmd' is submitted, so the
      // sequence of events are still preserved.
      m_rtBlas[0].update(m_rtContext, m_rtBlasInput[0], *m_rtScratchBuffer, cmd);

      // For TLAS, use PREFER_FAST_TRACE flag and perform only rebuilds,
      // https://developer.nvidia.com/blog/best-practices-for-using-nvidia-rtx-ray-tracing-updated/
      m_rtTlas->rebuild(m_rtContext, m_rtTlasInput, *m_rtScratchBuffer, cmd);
    }

    auto sdbg = m_dutil->DBG_SCOPE(cmd);

    float     view_aspect_ratio = m_viewSize.x / m_viewSize.y;
    glm::vec3 eye;
    glm::vec3 center;
    glm::vec3 up;
    CameraManip.getLookat(eye, center, up);

    // Update the uniform buffer containing frame info
    shaders::FrameInfo finfo{};
    const auto&        clip = CameraManip.getClipPlanes();
    finfo.view              = CameraManip.getMatrix();
    finfo.proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), view_aspect_ratio, clip.x, clip.y);
    finfo.proj[1][1] *= -1;
    finfo.projInv = glm::inverse(finfo.proj);
    finfo.viewInv = glm::inverse(finfo.view);
    finfo.camPos  = eye;
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(shaders::FrameInfo), &finfo);

    // Update the sky
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(nvvkhl_shaders::SimpleSkyParameters), &m_skyParams);

    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtSet.getSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0,
                            static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0, nullptr);

    m_pushConst.opacity         = m_settings.opacity;
    m_pushConst.refractiveIndex = m_settings.refractiveIndex;
    m_pushConst.density         = m_settings.density;
    m_pushConst.heightmapScale  = m_settings.heightmapScale;
    m_pushConst.maxDepth        = m_settings.maxDepth;
    m_pushConst.wireframeScale  = 1 << m_settings.subdivlevel;
    vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(shaders::PushConstant), &m_pushConst);

    const auto& regions = m_sbt.getRegions();
    const auto& size    = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, regions.data(), &regions[1], &regions[2], &regions[3], size.width, size.height, 1);
  }

private:
  void createScene()
  {
    float cubeHeight = 0.5f;
    m_meshes         = {
        nvh::createPlane(HEIGHTMAP_RESOLUTION / 32, 1.0F, 1.0F),
        nvh::createCube(1.0F, cubeHeight, 1.0F),
    };
    m_nodes = {
        nvh::Node{
            .mesh = 0,
        },
        nvh::Node{
            .mesh = 1,
        },
    };

    // Remove the top face of the cube and move down so it's flush with the
    // heightmap-displaced plane.
    m_meshes[1].triangles.pop_back();
    m_meshes[1].triangles.pop_back();
    for(auto& v : m_meshes[1].vertices)
    {
      v.p.y -= cubeHeight * 0.5f;
    }

    // Setting camera to see the scene
    CameraManip.setClipPlanes({0.01F, 100.0F});
    CameraManip.setLookat({0.5F, 0.2F, 1.0F}, {0.0F, -0.2F, 0.0F}, {0.0F, 1.0F, 0.0F});

    // Default Sky values
    m_skyParams = nvvkhl_shaders::initSimpleSkyParameters();
  }


  void createGbuffers(const glm::vec2& size)
  {
    vkDeviceWaitIdle(m_device);

    // Rendering image targets
    m_viewSize = size;
    m_gBuffer  = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                   VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)},
                                                   m_colorFormat, m_depthFormat);
  }

  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    auto* cmd = m_app->createTempCmdBuffer();
    m_bMeshes.resize(m_meshes.size());

    auto rt_usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                         | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex and Index per mesh
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      auto& m = m_bMeshes[i];

      // Adjust texture coordinates to land exactly on texel centers. This is
      // needed because heightmap_rtx samples the heightmap using GLSL's
      // texture() where pixel values are at texel centers, e.g. {0.5 / width,
      // 0.5 / height} samples pixel {0, 0}. However, nvh::createPlane()
      // produces texture coordinates in the range [0.0, 1.0] inclusive. Also
      // flip the Y coordinate to match the Imgui preview image.
      float scale  = (float(HEIGHTMAP_RESOLUTION) - 1.0f) / float(HEIGHTMAP_RESOLUTION);
      float offset = 0.5f / float(HEIGHTMAP_RESOLUTION);
      for(auto& v : m_meshes[i].vertices)
      {
        v.t   = v.t * scale + offset;
        v.t.y = 1.0f - v.t.y;
      }

      m.vertices = m_alloc->createBuffer(cmd, m_meshes[i].vertices, rt_usage_flag);
      m.indices  = m_alloc->createBuffer(cmd, m_meshes[i].triangles, rt_usage_flag);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.indices.buffer, i);
    }

    // Create the buffer of the current frame, changing at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(shaders::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    // Create the buffer of sky parameters, updated at each frame
    m_bSkyParams = m_alloc->createBuffer(sizeof(nvvkhl_shaders::SimpleSkyParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bSkyParams.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  void createHrtxPipeline()
  {
    // A callback must be provided for buffer allocation (this example uses nvpro_core's AllocVma).
    HrtxAllocatorCallbacks allocatorCallbacks{
        [](const VkBufferCreateInfo bufferCreateInfo, const VkMemoryPropertyFlags memoryProperties, void* userPtr) {
          auto alloc  = reinterpret_cast<nvvkhl::AllocVma*>(userPtr);
          auto result = new nvvk::Buffer();
          *result     = alloc->createBuffer(bufferCreateInfo, memoryProperties);
          return &result->buffer;  // return pointer to member
        },
        [](VkBuffer* bufferPtr, void* userPtr) {
          auto alloc = reinterpret_cast<nvvkhl::AllocVma*>(userPtr);
          // reconstruct from pointer to member
          auto nvvkBuffer = reinterpret_cast<nvvk::Buffer*>(reinterpret_cast<char*>(bufferPtr) - offsetof(nvvk::Buffer, buffer));
          alloc->destroy(*nvvkBuffer);
          delete nvvkBuffer;
        },
        m_alloc.get(),
    };

    // Create a HrtxPipeline object. This holds the shader and resources for baking
    HrtxPipelineCreate hrtxPipelineCreate{
        .physicalDevice      = m_app->getPhysicalDevice(),
        .device              = m_app->getDevice(),
        .allocator           = allocatorCallbacks,
        .instance            = VK_NULL_HANDLE,
        .getInstanceProcAddr = nullptr,
        .getDeviceProcAddr   = nullptr,
        .pipelineCache       = VK_NULL_HANDLE,
        .checkResultCallback = [](VkResult result) { nvvk::checkResult(result, "HRTX"); },
    };
    auto* cmd = m_app->createTempCmdBuffer();
    if(hrtxCreatePipeline(cmd, &hrtxPipelineCreate, &m_hrtxPipeline) != VK_SUCCESS)
    {
      LOGW("Warning: Failed to create HrtxPipeline. Raytracing heightmaps will not work.\n");
    }
    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  HrtxMap createHrtxMap(const VkAccelerationStructureGeometryKHR& geometry,
                        uint32_t                                  triangleCount,
                        const PrimitiveMeshVk&                    mesh,
                        const nvvk::Texture&                      texture,
                        VkCommandBuffer                           cmd)
  {
    // Barrier to make sure writes by the compute shader are visible when creating the micromap
    VkImageMemoryBarrier imageBarrier = nvvk::makeImageMemoryBarrier(texture.image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    hrtxBarrierFlags(nullptr, nullptr, nullptr, nullptr, &imageBarrier.newLayout);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &imageBarrier);

    HrtxMapCreate mapCreate{
        .triangles           = &geometry.geometry.triangles,
        .primitiveCount      = triangleCount,
        .textureCoordsBuffer = {.deviceAddress = nvvk::getBufferDeviceAddress(m_device, mesh.vertices.buffer)
                                                 + offsetof(nvh::PrimitiveVertex, t)},
        .textureCoordsFormat = VK_FORMAT_R32G32_SFLOAT,
        .textureCoordsStride = sizeof(nvh::PrimitiveVertex),
        .directionsBuffer    = {.deviceAddress = nvvk::getBufferDeviceAddress(m_device, mesh.vertices.buffer)
                                                 + offsetof(nvh::PrimitiveVertex, n)},
        .directionsFormat    = VK_FORMAT_R32G32B32_SFLOAT,
        .directionsStride    = sizeof(nvh::PrimitiveVertex),
        .heightmapImage      = texture.descriptor,
        .heightmapBias       = -m_settings.heightmapScale * 0.5f,
        .heightmapScale      = m_settings.heightmapScale,
        .subdivisionLevel    = static_cast<uint32_t>(m_settings.subdivlevel),
    };

    HrtxMap hrtxMap{};
    if(hrtxCmdCreateMap(cmd, m_hrtxPipeline, &mapCreate, &hrtxMap) != VK_SUCCESS)
    {
      LOGW("Warning: Failed to create HrtxMap for mesh %p. Raytracing heightmaps will not work.\n", &mesh);
    }
    return hrtxMap;
  }

  void destroyHrtxMaps()
  {
    m_staticCommandPool->destroy(m_cmdHrtxUpdate);
    m_cmdHrtxUpdate = VK_NULL_HANDLE;

    hrtxDestroyMap(m_hrtxMap);
    m_hrtxMap = nullptr;
  }

  //--------------------------------------------------------------------------------------------------
  // Build a bottom level acceleration structure (BLAS) for each mesh
  //
  void createBottomLevelAS(VkCommandBuffer cmd)
  {
    // Prepare to create one BLAS per mesh
    m_rtBlasInput.clear();
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      // Each BLAS has only one geometry input.
      assert(!m_meshes[i].vertices.empty());
      std::vector<rt::SimpleGeometryInput> simpleGeometryInputs{
          rt::SimpleGeometryInput{
              .triangleCount = static_cast<uint32_t>(m_meshes[i].triangles.size()),
              .maxVertex    = static_cast<uint32_t>(m_meshes[i].vertices.size()) - 1,  // Max. index one less than count
              .indexAddress = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[i].indices.buffer),
              .vertexAddress = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[i].vertices.buffer),
              .vertexStride  = sizeof(nvh::PrimitiveVertex),
          },
      };
      m_rtBlasInput.push_back(
          rt::createBlasInput(simpleGeometryInputs, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
                                                        | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR));
    }

    // Create a HrtxMap for the first mesh. Most nvpro_core command buffers are
    // temporary/one-shot. In this case the same command buffer can simply be
    // re-submitted to rebuild all HrtxMap objects.
    // Note: this records a command buffer with a refernce to just one of the
    // double buffered heightmaps.
    assert(!m_cmdHrtxUpdate);
    assert(!m_hrtxMap);
    m_cmdHrtxUpdate = m_staticCommandPool->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true, 0, nullptr);
    m_hrtxMap       = createHrtxMap(m_rtBlasInput[0].geometries[0], m_rtBlasInput[0].rangeInfos[0].primitiveCount,
                                    m_bMeshes[0], m_heightmap.height(), m_cmdHrtxUpdate);
    if(!m_hrtxMap)
    {
      LOGE("ERROR: createHrtxMap() failed");
      exit(1);
    }
    m_rtDisplacement = hrtxMapDesc(m_hrtxMap);

    // TODO: ideally this would be a secondary command buffer, added to 'cmd'
    // with vkCmdExecuteCommands() instead of being submitted immediately.
    m_staticCommandPool->submit(1, &m_cmdHrtxUpdate);

    // Apply the heightmap to the first mesh. The pNext pointer is reused for
    // build updates, so the object is stored in m_rtDisplacement.
    m_rtBlasInput[0].geometries[0].geometry.triangles.pNext = m_settings.enableDisplacement ? &m_rtDisplacement : nullptr;

    // Create the bottom level acceleration structures
    m_rtBlas.clear();
    for(auto& blasInput : m_rtBlasInput)
    {
      rt::AccelerationStructureSizes blasSizes(m_rtContext, blasInput);
      m_rtBlas.emplace_back(m_rtContext, rt::AccelerationStructure(m_rtContext, blasInput.type, *blasSizes, 0),
                            blasInput, *m_rtScratchBuffer, cmd);
    }
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS(VkCommandBuffer cmd)
  {
    // Update the cube's scale so that the heightmap cannot intersect it
    m_nodes[1].scale.y = m_settings.heightmapScale * 1.01f;

    VkGeometryInstanceFlagsKHR instanceFlags =
        VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR | VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(m_nodes.size());
    for(auto& node : m_nodes)
    {
      instances.push_back(VkAccelerationStructureInstanceKHR{
          .transform           = nvvk::toTransformMatrixKHR(node.localMatrix()),  // Position of the instance
          .instanceCustomIndex = static_cast<uint32_t>(node.mesh) & 0x00FFFFFF,   // gl_InstanceCustomIndexEXT
          .mask                = 0xFF,
          .instanceShaderBindingTableRecordOffset = 0,  // We will use the same hit group for all objects
          .flags                                  = instanceFlags & 0xFFU,
          .accelerationStructureReference         = m_rtBlas[node.mesh].address(),
      });
    };
    m_rtInstances = std::make_unique<rt::InstanceBuffer>(m_rtContext, instances, cmd);
    m_rtTlasInput = rt::createTlasInput(instances.size(), m_rtInstances->address(),
                                        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                            | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);

    // Create the top level acceleration structure
    rt::AccelerationStructureSizes tlasSizes(m_rtContext, m_rtTlasInput);
    m_rtTlas = std::make_unique<rt::BuiltAccelerationStructure>(
        m_rtContext, rt::AccelerationStructure(m_rtContext, m_rtTlasInput.type, *tlasSizes, 0), m_rtTlasInput,
        *m_rtScratchBuffer, cmd);
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    auto& p = m_rtPipe;
    auto& d = m_rtSet;
    p.plines.resize(1);

    // This descriptor set, holds the top level acceleration structure and the output image
    // Create Binding Set
    d.addBinding(BRtTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    d.addBinding(BRtOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d.addBinding(BRtFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d.addBinding(BRtSkyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d.initLayout();
    d.initPool(1);

    m_dutil->DBG_NAME(d.getLayout());
    m_dutil->DBG_NAME(d.getSet(0));

    // Creating all shaders
    enum StageIndices
    {
      eRaygen,
      eMiss,
      eClosestHit,
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.pName = "main";  // All the same entry point

    // Raygen
    m_rtShaderRgen  = ShaderModule(m_device, pathtrace_rgen);
    stage.module    = m_rtShaderRgen;
    stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eRaygen] = stage;
    m_dutil->setObjectName(stage.module, "Raygen");
    // Miss
    m_rtShaderRmiss = ShaderModule(m_device, pathtrace_rmiss);
    stage.module    = m_rtShaderRmiss;
    stage.stage     = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMiss]   = stage;
    m_dutil->setObjectName(stage.module, "Miss");
    // Hit Group - Closest Hit
    m_rtShaderRchit     = ShaderModule(m_device, pathtrace_rchit);
    stage.module        = m_rtShaderRchit;
    stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eClosestHit] = stage;
    m_dutil->setObjectName(stage.module, "Closest Hit");


    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group.anyHitShader       = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
    // Raygen
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    shader_groups.push_back(group);

    // Miss
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    shader_groups.push_back(group);

    // Closest hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    shader_groups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(shaders::PushConstant)};

    VkPipelineLayoutCreateInfo pipeline_layout_create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_create_info.pushConstantRangeCount = 1;
    pipeline_layout_create_info.pPushConstantRanges    = &push_constant;


    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = {d.getLayout()};  // , m_pContainer[eGraphic].dstLayout};
    pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(rt_desc_set_layouts.size());
    pipeline_layout_create_info.pSetLayouts                = rt_desc_set_layouts.data();
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout));
    m_dutil->DBG_NAME(p.layout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ray_pipeline_info.flags      = VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV;  // #MICROMESH
    ray_pipeline_info.stageCount = static_cast<uint32_t>(stages.size());                         // Stages are shaders
    ray_pipeline_info.pStages    = stages.data();
    ray_pipeline_info.groupCount = static_cast<uint32_t>(shader_groups.size());
    ray_pipeline_info.pGroups    = shader_groups.data();
    ray_pipeline_info.maxPipelineRayRecursionDepth = 10;  // Ray depth
    ray_pipeline_info.layout                       = p.layout;
    NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, (p.plines).data()));
    m_dutil->DBG_NAME(p.plines[0]);

    // Creating the SBT
    m_sbt.create(p.plines[0], ray_pipeline_info);
  }

  void writeRtDesc()
  {
    auto& d = m_rtSet;

    // Write to descriptors
    VkAccelerationStructureKHR tlas = *m_rtTlas;
    VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    desc_as_info.accelerationStructureCount = 1;
    desc_as_info.pAccelerationStructures    = &tlas;
    VkDescriptorImageInfo  image_info{{}, m_gBuffer->getColorImageView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dbi_sky{m_bSkyParams.buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(d.makeWrite(0, BRtTlas, &desc_as_info));
    writes.emplace_back(d.makeWrite(0, BRtOutImage, &image_info));
    writes.emplace_back(d.makeWrite(0, BRtFrameInfo, &dbi_unif));
    writes.emplace_back(d.makeWrite(0, BRtSkyParam, &dbi_sky));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void destroyResources()
  {
    vkDeviceWaitIdle(m_device);

    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    for(auto& m : m_bMeshes)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_alloc->destroy(m_bFrameInfo);
    m_alloc->destroy(m_bSkyParams);

    ImGui_ImplVulkan_RemoveTexture(m_heightmapImguiDesc);
    m_heightmap.destroy(*m_alloc);

    hrtxDestroyMap(m_hrtxMap);
    hrtxDestroyPipeline(m_hrtxPipeline);

    m_rtSet.deinit();
    m_gBuffer.reset();

    m_rtPipe.destroy(m_device);

    m_sbt.destroy();
    m_rtShaderRgen  = ShaderModule();
    m_rtShaderRmiss = ShaderModule();
    m_rtShaderRchit = ShaderModule();

    m_rtScratchBuffer.reset();
    m_rtBlas.clear();
    m_rtInstances.reset();
    m_rtTlas.reset();
  }

  nvvkhl::Application*               m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>   m_dutil;
  std::unique_ptr<nvvkhl::AllocVma>  m_alloc;
  std::unique_ptr<nvvk::CommandPool> m_staticCommandPool;

  glm::vec2                        m_viewSize    = {1, 1};
  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue                m_clearColor  = {{0.3F, 0.3F, 0.3F, 1.0F}};     // Clear color
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffer;                                      // G-Buffers: color + depth
  nvvkhl_shaders::SimpleSkyParameters m_skyParams{};

  // GPU scene buffers
  std::vector<PrimitiveMeshVk> m_bMeshes;
  nvvk::Buffer                 m_bFrameInfo;
  nvvk::Buffer                 m_bSkyParams;

  // Data and settings
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node>          m_nodes;

  // Raytracing pipeline
  nvvk::DescriptorSetContainer m_rtSet;                              // Descriptor set
  shaders::PushConstant        m_pushConst{};                        // Information sent to the shader
  VkPipelineLayout             m_pipelineLayout   = VK_NULL_HANDLE;  // The description of the pipeline
  VkPipeline                   m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
  int                          m_frame{0};

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  ShaderModule                                           m_rtShaderRgen;
  ShaderModule                                           m_rtShaderRmiss;
  ShaderModule                                           m_rtShaderRchit;
  nvvk::SBTWrapper                                       m_sbt;  // Shader binding table wrapper
  rt::Context                                            m_rtContext;
  std::unique_ptr<rt::ScratchBuffer>                     m_rtScratchBuffer;
  VkAccelerationStructureTrianglesDisplacementMicromapNV m_rtDisplacement;
  std::vector<rt::AccelerationStructureInput>            m_rtBlasInput;
  std::vector<rt::BuiltAccelerationStructure>            m_rtBlas;
  std::unique_ptr<rt::InstanceBuffer>                    m_rtInstances;
  rt::AccelerationStructureInput                         m_rtTlasInput;
  std::unique_ptr<rt::BuiltAccelerationStructure>        m_rtTlas;
  nvvkhl::PipelineContainer                              m_rtPipe;

  HrtxPipeline      m_hrtxPipeline{};
  HrtxMap           m_hrtxMap{};
  AnimatedHeightmap m_heightmap;
  VkDescriptorSet   m_heightmapImguiDesc = VK_NULL_HANDLE;
  VkCommandBuffer   m_cmdHrtxUpdate      = VK_NULL_HANDLE;
};

int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name                       = PROJECT_NAME " Example";
  spec.vSync                      = false;
  nvvk::ContextCreateInfo vkSetup = nvvk::ContextCreateInfo(false);  // #MICROMESH cannot have validation layers (crash)
  vkSetup.apiMajor                = 1;
  vkSetup.apiMinor                = 3;

  vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accel_feature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rt_pipeline_feature);  // To use vkCmdTraceRaysKHR
  vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
  vkSetup.addDeviceExtension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR rt_position_fetch{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};
  vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, false, &rt_position_fetch);

  // #MICROMESH
  static VkPhysicalDeviceOpacityMicromapFeaturesEXT mm_opacity_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT};
  static VkPhysicalDeviceDisplacementMicromapFeaturesNV mm_displacement_features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_FEATURES_NV};
  vkSetup.addDeviceExtension(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME, true, &mm_opacity_features);
  vkSetup.addDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME, true, &mm_displacement_features);

  // UI default docking
  spec.dockSetup = [](ImGuiID viewportID) {
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.2F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);
    ImGuiID heightmapID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.382F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Heightmap", heightmapID);
  };

  // Display extension
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  vkSetup.instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);

  // Creating the Vulkan context
  auto m_context = std::make_shared<nvvk::Context>();
  m_context->init(vkSetup);
  // Disable error messages introduced by micromesh
  m_context->ignoreDebugMessage(0x901f59ec);  // Unknown extension
  m_context->ignoreDebugMessage(0xdd73dbcf);  // Unknown structure
  m_context->ignoreDebugMessage(0xba164058);  // Unknown flag  vkGetAccelerationStructureBuildSizesKHR:
  m_context->ignoreDebugMessage(0x22d5bbdc);  // Unknown flag  vkCreateRayTracingPipelinesKHR
  m_context->ignoreDebugMessage(0x27112e51);  // Unknown flag  vkCreateBuffer
  m_context->ignoreDebugMessage(0x79de34d4);  // Unknown VK_NV_displacement_micromesh, VK_NV_opacity_micromesh

  // Application Vulkan setup
  spec.instance       = m_context->m_instance;
  spec.device         = m_context->m_device;
  spec.physicalDevice = m_context->m_physicalDevice;
  spec.queues.push_back({m_context->m_queueGCT.familyIndex, m_context->m_queueGCT.queueIndex, m_context->m_queueGCT.queue});
  spec.queues.push_back({m_context->m_queueC.familyIndex, m_context->m_queueC.queueIndex, m_context->m_queueC.queue});
  spec.queues.push_back({m_context->m_queueT.familyIndex, m_context->m_queueT.queueIndex, m_context->m_queueT.queue});

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // #MICROMESH
  if(!mm_opacity_features.micromap)
  {
    LOGE("ERROR: Micro-Mesh not supported");
    exit(1);
  }

  if(!mm_displacement_features.displacementMicromap)
  {
    LOGE("ERROR: Micro-Mesh displacement not supported");
    exit(1);
  }

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());         // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());  // Window title info
  app->addElement(std::make_shared<RaytracingSample>());


  app->run();

  vkDeviceWaitIdle(app->getDevice());
  app.reset();

  return test->errorCode();
}
