#version 330 core

// Fragment shader for computational graph visualization

// Input data from vertex shader
in vec3 fragmentPosition;
in vec3 fragmentNormal;
in vec2 fragmentUV;
in vec4 fragmentColor;
in float fragmentType;

// Output data
out vec4 color;

// Uniforms
uniform vec3 lightPosition;
uniform vec3 viewPosition;
uniform float time;

void main() {
    // Ambient lighting
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
    
    // Diffuse lighting
    vec3 norm = normalize(fragmentNormal);
    vec3 lightDir = normalize(lightPosition - fragmentPosition);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
    
    // Specular lighting
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPosition - fragmentPosition);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
    
    // Combine lighting with instance color
    vec3 result = (ambient + diffuse + specular) * fragmentColor.rgb;
    
    // Add subtle pulsing effect based on time and node type
    float pulse = 0.05 * sin(time * 2.0 + fragmentType * 0.5) + 0.95;
    result *= pulse;
    
    // Add rim lighting effect
    float rim = 1.0 - max(dot(viewDir, norm), 0.0);
    rim = smoothstep(0.4, 0.8, rim);
    result += rim * 0.3 * fragmentColor.rgb;
    
    // Output final color
    color = vec4(result, fragmentColor.a);
}
