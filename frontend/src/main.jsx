import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// The whole font package (every subset, each loaded only when the page needs it): the ₹ sign is in
// the latin-ext file, not the latin one.
import "@fontsource-variable/plus-jakarta-sans";

import App from "./App.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
