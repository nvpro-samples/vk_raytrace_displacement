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

#ifndef ANIMATE_HEIGHTMAP_H
#define ANIMATE_HEIGHTMAP_H

#ifdef __cplusplus
#include <glm/glm.hpp>
namespace shaders {
using mat4 = glm::mat4;
using vec4 = glm::vec4;
using vec3 = glm::vec3;
using vec2 = glm::vec2;
#endif  // __cplusplus

#define BINDING_ANIM_IMAGE_A_HEIGHT 0
#define BINDING_ANIM_IMAGE_B_HEIGHT 1
#define BINDING_ANIM_IMAGE_A_VELOCITY 2
#define BINDING_ANIM_IMAGE_B_VELOCITY 3

#define ANIMATION_WORKGROUP_SIZE 32

// Scale and offset values stored in the heightmap so that they're visible in
// the ImGui preview.
#define HEIGHTMAP_RANGE 0.1
#define HEIGHTMAP_OFFSET 0.5

struct AnimatePushConstants
{
  vec2     mouse;
  uint32_t writeToA;
  int      resolution;
  float    deltaTime;
};

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif  // ANIMATE_HEIGHTMAP_H
