#version 330 core

// Vertex shader for computational graph edges

// Input vertex data
layout(location = 0) in vec3 vertexPosition;

// Instance data
layout(location = 1) in vec3 instanceStart;
layout(location = 2) in vec3 instanceEnd;
layout(location = 3) in vec4 instanceColor;
layout(location = 4) in float instanceType;

// Output data to fragment shader
out vec4 fragmentColor;
out float fragmentType;
out float fragmentPosition;

// Uniforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;

void main() {
    // Calculate direction vector
    vec3 direction = instanceEnd - instanceStart;
    float length = length(direction);
    vec3 directionNormalized = normalize(direction);
    
    // Create a coordinate system for the edge
    vec3 up = vec3(0.0, 1.0, 0.0);
    if (abs(dot(directionNormalized, up)) > 0.99) {
        up = vec3(1.0, 0.0, 0.0);
    }
    vec3 right = normalize(cross(directionNormalized, up));
    up = normalize(cross(right, directionNormalized));
    
    // Calculate position along the edge
    float t = vertexPosition.z; // Use z coordinate as parameter along the edge
    vec3 position = mix(instanceStart, instanceEnd, t);
    
    // Add thickness to the edge
    float thickness = 0.01;
    position += (vertexPosition.x * right + vertexPosition.y * up) * thickness;
    
    // Add subtle animation based on time and edge type
    float animationFactor = sin(time * 2.0 + instanceType * 0.2 + t * 10.0) * 0.01;
    position += up * animationFactor;
    
    // Calculate final position
    vec4 worldPosition = model * vec4(position, 1.0);
    
    // Pass data to fragment shader
    fragmentColor = instanceColor;
    fragmentType = instanceType;
    fragmentPosition = t;
    
    // Output position
    gl_Position = projection * view * worldPosition;
}
