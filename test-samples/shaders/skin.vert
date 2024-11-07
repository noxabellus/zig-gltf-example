#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec4 aWeights;
layout (location = 3) in uvec4 aJoints;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vNormal;
out vec4 vWeights;
flat out uvec4 vJoints;

const uint MAX_JOINTS = 256u;

uniform mat4 skinMatrices [MAX_JOINTS];

void main() {
    vWeights = aWeights;
    vJoints = aJoints;

    mat4 skinMat
        = skinMatrices[aJoints.x] * aWeights.x
        + skinMatrices[aJoints.y] * aWeights.y
        + skinMatrices[aJoints.z] * aWeights.z
        + skinMatrices[aJoints.w] * aWeights.w;

    mat4 composedMat = projection * view * model * skinMat;

    vNormal = mat3(skinMat) * aNormal;
    gl_Position = composedMat * vec4(aPos, 1.0f);
}
