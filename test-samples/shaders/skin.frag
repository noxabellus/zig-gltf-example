#version 330 core

flat in uvec4 vJoints;
in vec4 vWeights;

out vec4 FragColor;

vec3 colors[6] = vec3[6](
    vec3(1.0f, 0.0f, 0.0f),
    vec3(0.0f, 1.0f, 0.0f),
    vec3(0.0f, 0.0f, 1.0f),
    vec3(1.0f, 1.0f, 0.0f),
    vec3(1.0f, 0.0f, 1.0f),
    vec3(0.0f, 1.0f, 1.0f)
);

void main() {
    vec3 colorX = colors[vJoints.x] * vWeights.x;
    vec3 colorY = colors[vJoints.y] * vWeights.y;
    FragColor = vec4(colorX + colorY + vec3(0.0), 1.0f);
}
