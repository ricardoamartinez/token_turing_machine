#version 330 core

// Vertex attributes
layout(location = 0) in vec3 position;      // Cube vertex positions
layout(location = 1) in vec3 normal;        // Cube vertex normals
layout(location = 2) in vec2 texCoord;      // Cube texture coordinates

// Instance attributes
layout(location = 3) in vec3 instancePos;   // Instance position
layout(location = 4) in vec3 instanceScale; // Instance scale
layout(location = 5) in vec4 instanceColor; // Instance color (RGBA)
layout(location = 6) in float instanceValue; // Instance data value (for color mapping)

// Uniforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform bool useColorMap;
uniform sampler1D colorMap;

// Outputs to fragment shader
out vec3 fragNormal;
out vec2 fragTexCoord;
out vec4 fragColor;
out float fragValue;

void main() {
    // Calculate model matrix for this instance
    mat4 instanceModel = model;
    
    // Apply instance position
    instanceModel[3][0] += instancePos.x;
    instanceModel[3][1] += instancePos.y;
    instanceModel[3][2] += instancePos.z;
    
    // Apply instance scale
    mat4 scaleMatrix = mat4(
        instanceScale.x, 0.0, 0.0, 0.0,
        0.0, instanceScale.y, 0.0, 0.0,
        0.0, 0.0, instanceScale.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    instanceModel = instanceModel * scaleMatrix;
    
    // Calculate final position
    gl_Position = projection * view * instanceModel * vec4(position, 1.0);
    
    // Transform normal to world space
    mat3 normalMatrix = transpose(inverse(mat3(instanceModel)));
    fragNormal = normalize(normalMatrix * normal);
    
    // Pass texture coordinates to fragment shader
    fragTexCoord = texCoord;
    
    // Pass color to fragment shader
    fragColor = instanceColor;
    
    // Pass value for color mapping
    fragValue = instanceValue;
}
