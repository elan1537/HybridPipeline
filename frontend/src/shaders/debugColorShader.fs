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
    // 3x2 그리드 레이아웃 (6분할)
    vec2 gridSize = vec2(3.0, 2.0);
    vec2 gridPos = floor(vUv * gridSize);
    vec2 localUv = fract(vUv * gridSize);
    
    vec4 color;
    
    // 상단 행 (y < 1.0) - Color 위주
    if (gridPos.y < 1.0) {
        if (gridPos.x < 1.0) {                                      // 좌상단: Local Color
            color = texture2D(localColorSampler, localUv);
        } else if (gridPos.x < 2.0) {                              // 중상단: Splat Color
            vec2 wsUv = vec2(1.0 - localUv.x, 1.0 - localUv.y);
            color = texture2D(wsColorSampler, wsUv);
        } else {                                                   // 우상단: Fusion Color
            // Fusion 로직: fusionColorShader.fs와 동일한 depth 비교
            vec2 wsUv = vec2(1.0 - localUv.x, 1.0 - localUv.y);
            
            vec4 localColor = texture2D(localColorSampler, localUv);
            float localDepth = texture2D(localDepthSampler, localUv).r;
            float d_8bit = floor(localDepth * 255.0 + 0.5) / 255.0;
            
            vec4 wsColor = texture2D(wsColorSampler, wsUv);
            float wsDepth = texture2D(wsDepthSampler, wsUv).r;
            
            bool useLocal = wsDepth > d_8bit;
            color = useLocal ? localColor : wsColor;
        }
    }
    // 하단 행 (y >= 1.0) - Depth 위주
    else {
        if (gridPos.x < 1.0) {                                      // 좌하단: Local Depth
            float depth = texture2D(localDepthSampler, localUv).r;
            float linearDepth = LinearizeDepth(depth);
            color = vec4(linearDepth, linearDepth, linearDepth, 1.0);
        } else if (gridPos.x < 2.0) {                              // 중하단: Splat Depth
            vec2 wsUv = vec2(1.0 - localUv.x, 1.0 - localUv.y);
            float depth = texture2D(wsDepthSampler, wsUv).r;
            color = vec4(depth, depth, depth, 1.0);
        } else {                                                   // 우하단: Fusion Depth
            // Fusion depth 시각화
            vec2 wsUv = vec2(1.0 - localUv.x, 1.0 - localUv.y);
            
            float localDepth = texture2D(localDepthSampler, localUv).r;
            float d_8bit = floor(localDepth * 255.0 + 0.5) / 255.0;
            float wsDepth = texture2D(wsDepthSampler, wsUv).r;
            
            bool useLocal = wsDepth > d_8bit;
            float fusionDepth = useLocal ? localDepth : wsDepth;
            
            // 컬러 맵핑으로 local/ws depth 구분
            vec3 depthColor;
            if (useLocal) {
                // Local depth: 푸른색 계열
                float linearDepth = LinearizeDepth(fusionDepth);
                depthColor = vec3(0.0, 0.5, 1.0) * linearDepth;
            } else {
                // WS depth: 주황색 계열  
                depthColor = vec3(1.0, 0.5, 0.0) * fusionDepth;
            }
            
            color = vec4(depthColor, 1.0);
        }
    }
    
    gl_FragColor = color;
}