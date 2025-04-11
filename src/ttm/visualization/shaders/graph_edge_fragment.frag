#version 330 core

// Fragment shader for computational graph edges

// Input data from vertex shader
in vec4 fragmentColor;
in float fragmentType;
in float fragmentPosition;

// Output data
out vec4 color;

// Uniforms
uniform float time;

void main() {
    // Base color from instance
    vec4 baseColor = fragmentColor;
    
    // Add flow animation for data flow edges
    if (fragmentType == 1.0) { // Data flow edge
        float flowSpeed = 0.5;
        float flowWidth = 0.1;
        float flowPosition = mod(time * flowSpeed, 1.0);
        float distanceFromFlow = min(
            abs(fragmentPosition - flowPosition),
            abs(fragmentPosition - flowPosition + 1.0)
        );
        
        // Create a glowing effect that moves along the edge
        float glowIntensity = smoothstep(flowWidth, 0.0, distanceFromFlow);
        baseColor.rgb += glowIntensity * 0.5;
    }
    
    // Add subtle pulsing effect based on time
    float pulse = 0.1 * sin(time * 1.5 + fragmentType * 0.3) + 0.9;
    baseColor.rgb *= pulse;
    
    // Output final color
    color = baseColor;
}
