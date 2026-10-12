import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// The whole font package (every subset, each loaded only when the page needs it): the ₹ sign is in
// the latin-ext file, not the latin one.
import "@fontsource-variable/plus-jakarta-sans";
// The shared styles first, so a page's own CSS (src/dashboard/dashboard.css) comes later and can
// adjust them.
import "./styles.css";

import App from "./App.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
