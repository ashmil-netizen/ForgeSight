/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#090e18",
        panel: "#111a28",
        accent: "#8b7cff",
        cyan: "#58d7e8",
      },
    },
  },
  plugins: [],
};
