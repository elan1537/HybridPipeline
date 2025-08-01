uniform sampler2D localColorSampler;
uniform sampler2D localDepthSampler;
uniform sampler2D wsColorSampler;
uniform sampler2D wsDepthSampler;

float near = 0.3;
float far = 1000.0;

float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));	
}

varying vec2 vUv;

void main() {
    vec2 gridPos = floor(vUv * 2.0); // 0 또는 1
    vec2 localUv = fract(vUv * 2.0); // 0~1 범위로 정규화
    
    vec4 color;
    
    if (gridPos.x < 0.5 && gridPos.y < 0.5) {                       // 좌상단: Local Color
        color = texture2D(localColorSampler, localUv);
    } else if (gridPos.x >= 0.5 && gridPos.y < 0.5) {               // 우상단: Local Depth
        float depth = texture2D(localDepthSampler, localUv).r;
        float linearDepth = LinearizeDepth(depth);
        color = vec4(linearDepth, linearDepth, linearDepth, 1.0);
    } else if (gridPos.x < 0.5 && gridPos.y >= 0.5) {               // 좌하단: Splat Color
        vec2 wsUv = vec2(localUv.x, 1.0 - localUv.y);
        color = texture2D(wsColorSampler, wsUv);
    } else {                                                        // 우하단: Splat Depth
        vec2 wsUv = vec2(localUv.x, 1.0 - localUv.y);
        float depth = texture2D(wsDepthSampler, wsUv).r;

        color = vec4(depth, depth, depth, 1.0);
    }
    
    gl_FragColor = color;
}