#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in uvec4 aJoints;
layout (location = 2) in vec4 aWeights;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

flat out uvec4 vJoints;
out vec4 vWeights;

const uint MAX_JOINTS = 256u;

uniform mat4 skinMatrices [MAX_JOINTS];

void main() {
    vJoints = aJoints;
    vWeights = aWeights;

    mat4 skinMat
        = skinMatrices[aJoints.x] * aWeights.x
        + skinMatrices[aJoints.y] * aWeights.y
        + skinMatrices[aJoints.z] * aWeights.z
        + skinMatrices[aJoints.w] * aWeights.w;

    mat4 composedMat = projection * view * model * skinMat;

    gl_Position = composedMat * vec4(aPos, 1.0f);
}
