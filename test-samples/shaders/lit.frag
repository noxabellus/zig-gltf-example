#version 330 core

uniform vec3 color;
uniform vec3 lightDir;

in vec3 vNormal;
out vec4 FragColor;

void main() {
    float intensity = max(dot(vNormal, normalize(lightDir)), 0.04);
    FragColor = vec4(color * intensity, 1.0f);
}
