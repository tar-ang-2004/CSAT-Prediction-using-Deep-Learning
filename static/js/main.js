// Main JavaScript for DeepCSAT Web Application

// Dark mode toggle functionality
class DarkModeToggle {
    constructor() {
        this.isDark = true; // Default dark mode
        this.init();
    }

    init() {
        // Check localStorage for saved preference
        const saved = localStorage.getItem('darkMode');
        if (saved !== null) {
            this.isDark = saved === 'true';
        }
        this.apply();
    }

    toggle() {
        this.isDark = !this.isDark;
        this.apply();
        localStorage.setItem('darkMode', this.isDark);
    }

    apply() {
        if (this.isDark) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }
}

// Initialize dark mode
const darkMode = new DarkModeToggle();

// Smooth scroll to top button
class ScrollToTop {
    constructor() {
        this.button = this.createButton();
        this.init();
    }

    createButton() {
        const btn = document.createElement('button');
        btn.innerHTML = '<i class="fas fa-arrow-up"></i>';
        btn.className = 'fixed bottom-8 right-8 w-12 h-12 bg-gradient-to-br from-primary-500 to-accent-500 rounded-full shadow-2xl opacity-0 pointer-events-none transition-all duration-300 z-50 hover:scale-110';
        btn.id = 'scroll-to-top';
        btn.setAttribute('aria-label', 'Scroll to top');
        document.body.appendChild(btn);
        return btn;
    }

    init() {
        // Show/hide button on scroll
        window.addEventListener('scroll', () => {
            if (window.pageYOffset > 300) {
                this.button.classList.remove('opacity-0', 'pointer-events-none');
            } else {
                this.button.classList.add('opacity-0', 'pointer-events-none');
            }
        });

        // Scroll to top on click
        this.button.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }
}

// Initialize scroll to top
const scrollToTop = new ScrollToTop();

// Intersection Observer for scroll animations
class ScrollAnimations {
    constructor() {
        this.observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        this.init();
    }

    init() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-fade-in-up');
                    observer.unobserve(entry.target);
                }
            });
        }, this.observerOptions);

        // Observe all elements with animate-on-scroll class
        document.querySelectorAll('.animate-on-scroll').forEach(el => {
            observer.observe(el);
        });
    }
}

// Initialize scroll animations on DOM load
document.addEventListener('DOMContentLoaded', () => {
    new ScrollAnimations();
});

// Form validation helper
class FormValidator {
    static validateNumber(value, min = null, max = null) {
        const num = parseFloat(value);
        if (isNaN(num)) return false;
        if (min !== null && num < min) return false;
        if (max !== null && num > max) return false;
        return true;
    }

    static showError(input, message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'text-red-400 text-sm mt-1 animate-fade-in';
        errorDiv.textContent = message;
        
        // Remove existing error
        const existing = input.parentElement.querySelector('.text-red-400');
        if (existing) existing.remove();
        
        input.parentElement.appendChild(errorDiv);
        input.classList.add('border-red-500');
    }

    static clearError(input) {
        const error = input.parentElement.querySelector('.text-red-400');
        if (error) error.remove();
        input.classList.remove('border-red-500');
    }
}

// API Helper class
class APIHelper {
    static async post(url, data) {
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    static async get(url) {
        try {
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }
}

// Loading spinner helper
class LoadingSpinner {
    static show(element) {
        element.innerHTML = `
            <div class="flex items-center justify-center py-8">
                <div class="spinner"></div>
            </div>
        `;
    }

    static hide(element) {
        element.innerHTML = '';
    }
}

// Toast notification system
class ToastNotification {
    static show(message, type = 'info', duration = 3000) {
        const colors = {
            'success': 'from-green-500 to-emerald-500',
            'error': 'from-red-500 to-rose-500',
            'info': 'from-blue-500 to-cyan-500',
            'warning': 'from-yellow-500 to-orange-500'
        };

        const icons = {
            'success': 'check-circle',
            'error': 'exclamation-circle',
            'info': 'info-circle',
            'warning': 'exclamation-triangle'
        };

        const toast = document.createElement('div');
        toast.className = `fixed top-24 right-6 px-6 py-4 bg-gradient-to-r ${colors[type]} rounded-xl shadow-2xl z-50 animate-slide-in-right max-w-md`;
        toast.innerHTML = `
            <div class="flex items-center space-x-3">
                <i class="fas fa-${icons[type]} text-white text-xl"></i>
                <span class="text-white font-medium">${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="ml-4 text-white hover:text-gray-200">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100px)';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
}

// Performance monitoring
class PerformanceMonitor {
    static measureExecutionTime(func, label) {
        const start = performance.now();
        const result = func();
        const end = performance.now();
        console.log(`${label}: ${(end - start).toFixed(2)}ms`);
        return result;
    }

    static async measureAsyncExecutionTime(func, label) {
        const start = performance.now();
        const result = await func();
        const end = performance.now();
        console.log(`${label}: ${(end - start).toFixed(2)}ms`);
        return result;
    }
}

// Local storage helper
class StorageHelper {
    static set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (error) {
            console.error('Storage error:', error);
            return false;
        }
    }

    static get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.error('Storage error:', error);
            return defaultValue;
        }
    }

    static remove(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (error) {
            console.error('Storage error:', error);
            return false;
        }
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        DarkModeToggle,
        FormValidator,
        APIHelper,
        LoadingSpinner,
        ToastNotification,
        PerformanceMonitor,
        StorageHelper
    };
}
