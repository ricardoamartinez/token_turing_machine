#version 330 core

// Vertex shader for computational graph visualization

// Input vertex data
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in vec2 vertexUV;

// Instance data
layout(location = 3) in vec3 instancePosition;
layout(location = 4) in vec3 instanceScale;
layout(location = 5) in vec4 instanceColor;
layout(location = 6) in float instanceType;

// Output data to fragment shader
out vec3 fragmentPosition;
out vec3 fragmentNormal;
out vec2 fragmentUV;
out vec4 fragmentColor;
out float fragmentType;

// Uniforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;

void main() {
    // Calculate instance model matrix
    mat4 instanceModel = mat4(1.0);
    
    // Scale
    instanceModel[0][0] = instanceScale.x;
    instanceModel[1][1] = instanceScale.y;
    instanceModel[2][2] = instanceScale.z;
    
    // Translate
    instanceModel[3][0] = instancePosition.x;
    instanceModel[3][1] = instancePosition.y;
    instanceModel[3][2] = instancePosition.z;
    
    // Apply subtle animation based on time and instance type
    float animationFactor = sin(time * 0.5 + instanceType * 0.1) * 0.05;
    instanceModel[3][1] += animationFactor;
    
    // Calculate position
    vec4 worldPosition = model * instanceModel * vec4(vertexPosition, 1.0);
    
    // Pass data to fragment shader
    fragmentPosition = worldPosition.xyz;
    fragmentNormal = mat3(transpose(inverse(model * instanceModel))) * vertexNormal;
    fragmentUV = vertexUV;
    fragmentColor = instanceColor;
    fragmentType = instanceType;
    
    // Output position
    gl_Position = projection * view * worldPosition;
}
