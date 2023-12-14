/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#ifdef __cplusplus
#include <glm/glm.hpp>
namespace shaders {
using mat4 = glm::mat4;
using vec4 = glm::vec4;
using vec3 = glm::vec3;
using vec2 = glm::vec2;
#endif  // __cplusplus

struct PushConstant
{
  float opacity;
  float refractiveIndex;
  float density;
  float heightmapScale;
  int   maxDepth;
  int   wireframeScale;
};

struct FrameInfo
{
  mat4 proj;
  mat4 view;
  mat4 projInv;
  mat4 viewInv;
  vec3 camPos;
};

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif  // HOST_DEVICE_H
