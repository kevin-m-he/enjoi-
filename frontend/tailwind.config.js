/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          900: '#0c0a14',
          800: '#171327',
          700: '#221b38',
        },
      },
      boxShadow: {
        glow: '0 0 24px rgba(236, 72, 153, 0.25)',
      },
    },
  },
  plugins: [],
};
