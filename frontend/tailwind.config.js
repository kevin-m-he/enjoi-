/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // --- vibrant pop accents ---
        cyan: {
          DEFAULT: '#00E5FF',
          400: '#22D3EE',
          500: '#00E5FF',
          600: '#06B6CC',
        },
        pink: {
          DEFAULT: '#FF2D95',
          400: '#FF5AAE',
          500: '#FF2D95',
          600: '#FF1FA2',
          700: '#D81B7E',
        },
        // --- The Great Wave / Kanagawa palette ---
        prussian: {
          DEFAULT: '#1B3A5B',
          900: '#0B2C4D',
          800: '#123450',
          700: '#1B3A5B',
          600: '#274C73',
        },
        foam: {
          DEFAULT: '#F4ECD8',
          50: '#FBF7EC',
          100: '#F4ECD8',
        },
        washi: {
          DEFAULT: '#F4ECD8',
          50: '#FBF7EC',
          100: '#F4ECD8',
          200: '#E9DEC2',
          300: '#DCCEA8',
        },
        ink: {
          DEFAULT: '#0B0B0C',
          900: '#0B0B0C',
          800: '#161617',
        },
      },
      fontFamily: {
        display: [
          'Archivo',
          'Arial Black',
          'Inter',
          'Segoe UI',
          'Microsoft YaHei',
          'system-ui',
          'sans-serif',
        ],
        body: ['Inter', 'Segoe UI', 'Microsoft YaHei', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        brutal: '6px 6px 0 0 #0B0B0C',
        'brutal-sm': '3px 3px 0 0 #0B0B0C',
        'brutal-lg': '10px 10px 0 0 #0B0B0C',
        'brutal-cyan': '6px 6px 0 0 #00E5FF',
        'brutal-pink': '6px 6px 0 0 #FF2D95',
      },
      borderWidth: {
        3: '3px',
        4: '4px',
        5: '5px',
      },
      borderRadius: {
        brutal: '4px',
      },
      keyframes: {
        'wave-march': {
          '0%': { backgroundPosition: '0 0' },
          '100%': { backgroundPosition: '48px 0' },
        },
        'foam-bob': {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-3px)' },
        },
      },
      animation: {
        'wave-march': 'wave-march 1s linear infinite',
        'foam-bob': 'foam-bob 1.6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
