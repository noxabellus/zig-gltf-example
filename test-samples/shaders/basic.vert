#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 vNormal;

uniform mat4 modelToClip;

void main() {
    vNormal = aNormal;
    gl_Position = vec4(aPos, 1.0) * modelToClip;
}
