#version 330 core

// Inputs from vertex shader
in vec3 fragNormal;
in vec2 fragTexCoord;
in vec4 fragColor;
in float fragValue;

// Uniforms
uniform bool useColorMap;
uniform sampler1D colorMap;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform bool enableLighting;

// Output
out vec4 finalColor;

void main() {
    // Base color
    vec4 color;
    
    // Use color map if enabled, otherwise use instance color
    if (useColorMap) {
        color = texture(colorMap, fragValue);
    } else {
        color = fragColor;
    }
    
    // Apply lighting if enabled
    if (enableLighting) {
        // Ambient
        float ambientStrength = 0.3;
        vec3 ambient = ambientStrength * color.rgb;
        
        // Diffuse
        vec3 lightDir = normalize(lightPos - vec3(fragTexCoord, 0.0));
        float diff = max(dot(fragNormal, lightDir), 0.0);
        vec3 diffuse = diff * color.rgb;
        
        // Specular
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - vec3(fragTexCoord, 0.0));
        vec3 reflectDir = reflect(-lightDir, fragNormal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
        
        // Combine
        vec3 result = ambient + diffuse + specular;
        finalColor = vec4(result, color.a);
    } else {
        finalColor = color;
    }
}
