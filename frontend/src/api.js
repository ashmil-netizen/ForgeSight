import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api",
  timeout: 15000,
});

export function readableError(error) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg).join("; ");
  }
  if (error?.code === "ECONNABORTED") {
    return "The API request timed out. Check that the backend is running.";
  }
  if (error?.request) {
    return "Could not reach the API. Start the backend and try again.";
  }
  return error?.message || "Unexpected error. Please try again.";
}

export async function getModelInfo() {
  return (await api.get("/model-info")).data;
}

export async function getHealth() {
  return (await api.get("/health")).data;
}

export async function getPredictions(limit = 50) {
  return (await api.get("/predictions", { params: { limit } })).data;
}

export async function createPrediction(payload) {
  return (await api.post("/predict", payload)).data;
}
