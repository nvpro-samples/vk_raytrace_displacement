/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_NV_displacement_micromap : require
#extension GL_EXT_ray_tracing_position_fetch : require

#include "device_host.h"
#include "animate_heightmap.h"
#include "dh_bindings.h"
#include "payload.h"
#include "nvvkhl/shaders/dh_sky.h"

// Barycentric coordinates of hit location relative to triangle vertices
hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;
layout(set = 0, binding = BRtTlas ) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = BRtFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = BRtSkyParam) uniform SkyInfo_ { SimpleSkyParameters skyInfo; };
layout(push_constant) uniform RtxPushConstant_ { PushConstant pc; };
// clang-format on

// Return true if there is no occluder, meaning that the light is visible from P toward L
bool shadowRay(vec3 P, vec3 L)
{
  const uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
  HitPayload savedP = payload;
  traceRayEXT(topLevelAS, rayFlags, 0xFF, 0, 0, 0, P, 0.0001, L, 100.0, 0);
  bool visible = (payload.depth == MISS_DEPTH);
  payload      = savedP;
  return visible;
}

float fresnelSchlickApprox(vec3 incident, vec3 normal, float ior)
{
  float r0 = (ior - 1.0) / (ior + 1.0);
  r0 *= r0;
  float cosX = -dot(normal, incident);
  if(ior > 1.0)
  {
    float sinT2 = ior * ior * (1.0 - cosX * cosX);
    // Total internal reflection
    if(sinT2 > 1.0)
      return 1.0;
    cosX = sqrt(1.0 - sinT2);
  }
  float x   = 1.0 - cosX;
  float ret = r0 + (1.0 - r0) * x * x * x * x * x;
  return ret;
}

// utility for temperature
float fade(float low, float high, float value)
{
  float mid   = (low + high) * 0.5;
  float range = (high - low) * 0.5;
  float x     = 1.0 - clamp(abs(mid - value) / range, 0.0, 1.0);
  return smoothstep(0.0, 1.0, x);
}

// Return a cold-hot color based on intensity [0-1]
vec3 temperature(float intensity)
{
  const vec3 water = vec3(0.0, 0.0, 0.5);
  const vec3 sand  = vec3(0.8, 0.7, 0.4);
  const vec3 green = vec3(0.1, 0.4, 0.1);
  const vec3 rock  = vec3(0.4, 0.4, 0.4);
  const vec3 snow  = vec3(1.0, 1.0, 1.0);


  vec3 color = (fade(-0.25, 0.25, intensity) * water   //
                + fade(0.0, 0.5, intensity) * sand     //
                + fade(0.25, 0.75, intensity) * green  //
                + fade(0.5, 1.0, intensity) * rock     //
                + smoothstep(0.75, 1.0, intensity) * snow);
  return color;
}

vec2 baseToMicro(vec2 barycentrics[3], vec2 p)
{
  vec2  ap   = p - barycentrics[0];
  vec2  ab   = barycentrics[1] - barycentrics[0];
  vec2  ac   = barycentrics[2] - barycentrics[0];
  float rdet = 1.f / (ab.x * ac.y - ab.y * ac.x);
  return vec2(ap.x * ac.y - ap.y * ac.x, ap.y * ab.x - ap.x * ab.y) * rdet;
}

void wireframe(inout float wire, float width, vec3 bary)
{
  float minBary = min(bary.x, min(bary.y, bary.z));
  wire          = min(wire, smoothstep(width, width + 0.002F, minBary));
}

void main()
{
  // We hit our max depth
  if(payload.depth >= pc.maxDepth)
  {
    return;
  }

  // NOTE: may not access gl_HitMicroTriangleVertexPositionsNV or
  // gl_HitMicroTriangleVertexBarycentricsNV for non-micromesh triangles.
  bool isMicromesh = gl_HitKindEXT == gl_HitKindFrontFacingMicroTriangleNV || gl_HitKindEXT == gl_HitKindBackFacingMicroTriangleNV;
  bool isFront = gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT || gl_HitKindEXT == gl_HitKindFrontFacingMicroTriangleNV;

  vec3 wPos   = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
  vec3 wDir   = normalize(gl_WorldRayDirectionEXT);
  vec3 wEye   = -wDir;
  vec3 wLight = normalize(skyInfo.directionToLight);
  vec3 wNorm  = isMicromesh ?
                    normalize(cross(gl_HitMicroTriangleVertexPositionsNV[2] - gl_HitMicroTriangleVertexPositionsNV[0],
                                    gl_HitMicroTriangleVertexPositionsNV[2] - gl_HitMicroTriangleVertexPositionsNV[1])) :
                    normalize(cross(gl_HitTriangleVertexPositionsEXT[2] - gl_HitTriangleVertexPositionsEXT[0],
                                    gl_HitTriangleVertexPositionsEXT[2] - gl_HitTriangleVertexPositionsEXT[1]));

  float height = (wPos.y / pc.heightmapScale) * 2.0f + 0.5f;

  wNorm = isFront ? wNorm : -wNorm;

  vec3 albedo = vec3(0.2, 0.2, 0.8);

  // Add wireframe
  float opacity = pc.opacity;
  if(isMicromesh)
  {
    // Color based on the height
    albedo = temperature(height);

    float wire = 1.0;
    const vec2 microBary2 = baseToMicro(gl_HitMicroTriangleVertexBarycentricsNV, attribs);
    const vec3 microBary  = vec3(1.0F - microBary2.x - microBary2.y, microBary2.xy);
    wireframe(wire, 0.002F * pc.wireframeScale, microBary);

    const vec3 baseBary = vec3(1.0 - attribs.x - attribs.y, attribs.xy);
    wireframe(wire, 0.008F, baseBary);

    const vec3 wireColor = vec3(0.3F, 0.3F, 0.3F);
    albedo = mix(wireColor, albedo, wire);
    opacity = mix(1.0, opacity, wire);
  }

  float ior              = isFront ? (1.0 / pc.refractiveIndex) : pc.refractiveIndex;
  float reflectivity     = fresnelSchlickApprox(wDir, wNorm, ior);
  vec3  reflectionWeight = payload.weight * reflectivity;
  vec3  refractionWeight = payload.weight * (1.0 - reflectivity);
  int   newDepth         = payload.depth + 1;

  if(isFront)
  {
    // Add light contribution unless in shadow
    bool visible = shadowRay(wPos, wLight);
    if(visible)
    {
      float diffuse  = clamp(dot(wNorm, wLight), 0.0, 1.0);
      payload.color += payload.weight * albedo * diffuse * opacity;
    }
  }
  else
  {
    // Absorption - Beer's law
    vec3 density    = vec3(0.8, 0.8, 0.4);
    vec3 absorption = exp(-density * pc.density * gl_HitTEXT);
    reflectionWeight *= absorption;
    refractionWeight *= absorption;
  }
  refractionWeight *= (1.0 - opacity);

  // Note: the following follows both sides of the branch, which is slow

  // Reflection
  if(max(max(reflectionWeight.x, reflectionWeight.y), reflectionWeight.z) > 0.01)
  {
    vec3 reflectDir = reflect(wDir, wNorm);
    payload.weight  = reflectionWeight;
    payload.depth   = newDepth;
    traceRayEXT(topLevelAS, gl_RayFlagsCullBackFacingTrianglesEXT, 0xFF, 0, 0, 0, wPos, 0.0001, reflectDir, 100.0, 0);
  }

  // Refraction
  if(max(max(refractionWeight.x, refractionWeight.y), refractionWeight.z) > 0.01)
  {
    vec3 refractDir = refract(wDir, wNorm, ior);
    payload.weight  = refractionWeight;
    payload.depth   = newDepth;
    traceRayEXT(topLevelAS, gl_RayFlagsCullBackFacingTrianglesEXT, 0xFF, 0, 0, 0, wPos, 0.0001, refractDir, 100.0, 0);
  }
}
