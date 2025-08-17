// ui-controller.ts
export class UIController {
    private isUIVisible: boolean = true;
    private uiContainer: HTMLElement | null = null;
    private hideTimeout: number | null = null;
    private readonly hideDelay = 3000; // 3초 후 UI 숨김

    constructor() {
        // Ensure DOM is ready before initializing
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.init();
            });
        } else {
            this.init();
        }
    }

    private init() {
        this.setupUI();
        this.setupEventListeners();
    }

    private setupUI() {
        this.uiContainer = document.getElementById('fps-overlay');
        if (!this.uiContainer) {
            console.warn('UI container not found');
            return;
        }

        // 마우스 이동 시 UI 표시
        this.setupAutoHide();
    }

    private setupEventListeners() {
        // 키보드 단축키 - Use capture phase to ensure we get events first
        document.addEventListener('keydown', (event) => {
            switch (event.key.toLowerCase()) {
                case 'h':
                    event.preventDefault();
                    event.stopPropagation();
                    this.toggleUI();
                    break;
                case 'escape':
                    event.preventDefault();
                    event.stopPropagation();
                    this.showUI();
                    break;
            }
        }, true); // Use capture phase

        // 더블클릭으로 UI 토글
        document.addEventListener('dblclick', (event) => {
            // UI 영역이 아닌 곳을 더블클릭했을 때만
            if (!this.uiContainer?.contains(event.target as Node)) {
                this.toggleUI();
            }
        });
    }

    private setupAutoHide() {
        let mouseMoveTimeout: number | null = null;

        document.addEventListener('mousemove', () => {
            this.showUI();
            
            // 기존 타이머 취소
            if (mouseMoveTimeout) {
                clearTimeout(mouseMoveTimeout);
            }
            
            // 새 타이머 설정
            mouseMoveTimeout = window.setTimeout(() => {
                this.hideUI();
            }, this.hideDelay);
        });

        // 마우스가 UI 영역에 있을 때는 숨기지 않음
        this.uiContainer?.addEventListener('mouseenter', () => {
            if (mouseMoveTimeout) {
                clearTimeout(mouseMoveTimeout);
                mouseMoveTimeout = null;
            }
        });

        this.uiContainer?.addEventListener('mouseleave', () => {
            mouseMoveTimeout = window.setTimeout(() => {
                this.hideUI();
            }, this.hideDelay);
        });
    }

    toggleUI() {
        if (this.isUIVisible) {
            this.hideUI();
        } else {
            this.showUI();
        }
    }

    showUI() {
        if (!this.uiContainer) return;
        
        this.isUIVisible = true;
        this.uiContainer.style.opacity = '1';
        this.uiContainer.style.pointerEvents = 'auto';
        
        // 커서 표시
        document.body.style.cursor = 'auto';
    }

    hideUI() {
        if (!this.uiContainer) return;
        
        this.isUIVisible = false;
        this.uiContainer.style.opacity = '0';
        this.uiContainer.style.pointerEvents = 'none';
        
        // 커서 숨김
        document.body.style.cursor = 'none';
    }

    isVisible(): boolean {
        return this.isUIVisible;
    }

    // UI 요소에 이벤트 리스너 추가하는 헬퍼 메서드
    addControlListener(elementId: string, event: string, handler: () => void) {
        const element = document.getElementById(elementId);
        if (element) {
            element.addEventListener(event, handler);
        } else {
            console.warn(`Element with ID '${elementId}' not found`);
        }
    }

    // UI 요소 값 설정하는 헬퍼 메서드
    setElementText(elementId: string, text: string) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = text;
        }
    }

    // 체크박스 상태 설정
    setCheckboxState(elementId: string, checked: boolean) {
        const element = document.getElementById(elementId) as HTMLInputElement;
        if (element) {
            element.checked = checked;
        }
    }

    // 라디오 버튼 상태 설정
    setRadioState(elementId: string, checked: boolean) {
        const element = document.getElementById(elementId) as HTMLInputElement;
        if (element) {
            element.checked = checked;
        }
    }

    // 셀렉트 박스 값 설정
    setSelectValue(elementId: string, value: string) {
        const element = document.getElementById(elementId) as HTMLSelectElement;
        if (element) {
            element.value = value;
        }
    }
}

// 전역 인스턴스
export const uiController = new UIController();