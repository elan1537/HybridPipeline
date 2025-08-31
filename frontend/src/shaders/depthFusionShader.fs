uniform sampler2D localColorSampler;
uniform sampler2D localDepthSampler;
uniform sampler2D wsColorSampler;
uniform sampler2D wsDepthSampler;
uniform bool wsFlipX;
uniform bool fusionFlipX;
varying vec2 vUv;

float near = 0.3;
float far = 100.0;

float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));	
}

void main() {
    // 기본 UV 계산
    vec2 currentUv = vUv;

    // fusionFlipX가 true이면 UV를 플립해서 모든 계산을 다시 수행
    if (fusionFlipX) {
        currentUv.x = 1.0 - currentUv.x;
    }

    vec2 localUv = currentUv;
    vec2 wsUv = vec2(currentUv.x, 1.0 - currentUv.y);

    // wsFlipX가 true일 때만 wsUv의 X축을 추가로 flip
    if (wsFlipX) {
        wsUv.x = 1.0 - wsUv.x;
    }

    // Depth 값들 가져오기
    float localDepth = texture2D(localDepthSampler, localUv).r;
    float d_8bit = floor(localDepth * 255.0 + 0.5) / 255.0;
    float wsDepth = texture2D(wsDepthSampler, wsUv).r;

    // Fusion 로직: wsDepth > d_8bit이면 local 사용
    bool useLocal = wsDepth > d_8bit;

    // 컬러 맵핑으로 local/gaussian depth 구분
    vec3 depthColor;
    if (useLocal) {
        // Local depth: 푸른색 계열
        float linearDepth = LinearizeDepth(localDepth);
        // 0.1-1.0 범위로 정규화하여 시각화 개선
        float normalizedDepth = clamp(linearDepth / 50.0, 0.1, 1.0);
        depthColor = vec3(0.0, 0.8, 1.0) * normalizedDepth;
    } else {
        // Gaussian depth: 주황색 계열  
        // wsDepth는 이미 0-1 범위이므로 직접 사용
        float normalizedDepth = clamp(wsDepth, 0.1, 1.0);
        depthColor = vec3(1.0, 0.8, 0.0) * normalizedDepth;
    }
    
    gl_FragColor = vec4(depthColor, 1.0);
}