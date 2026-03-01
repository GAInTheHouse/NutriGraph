# NutriGraph

A nutrition estimation and tracking application built with Streamlit. This app provides interfaces for both diners (consumers) and restaurants to estimate and manage nutritional information.

## Features

### Diner View
- **Dish Search**: Search for dishes by name with optional restaurant context
- **Nutrition Estimation**: Get estimated calories, protein, carbs, and fat
- **Ingredient Breakdown**: View estimated ingredients for any dish
- **Personalized Tracking**: Track daily nutrition (coming soon)
- **Feedback System**: Report inaccurate estimates

### Restaurant View
- **Dish Builder**: Create dishes with custom ingredient lists
- **Nutrition Profile Generator**: Generate accurate nutrition profiles
- **Catalog Management**: Manage your menu's nutrition data
- **Export**: Download catalog as CSV

## Project Structure

```
NutriGraph/
├── app.py                    # Streamlit entrypoint
├── src/
│   ├── core/
│   │   ├── api_client.py     # Backend API client (mock)
│   │   ├── config.py         # Settings and constants
│   │   └── models.py         # Pydantic data models
│   ├── ui/
│   │   ├── components.py     # Shared UI widgets
│   │   ├── diner.py          # Diner tab interface
│   │   └── restaurant.py     # Restaurant tab interface
│   ├── backend/
│   │   └── retrieval_server.py  # FastAPI retrieval backend (ingredients → vector DB)
│   └── ml/
│       └── extract_ingredients.py  # Image → ingredient list via Gemini
├── environment.yml           # Conda environment file
├── requirements.txt          # Python dependencies (pip)
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## Quick Start

### 1. Create Conda Environment

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate nutrigraph
```

**Alternative:** Create manually and install with pip:

```bash
conda create -n nutrigraph python=3.11
conda activate nutrigraph
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env    # macOS/Linux
copy .env.example .env  # Windows

# Edit .env with your settings (optional)
```

### 3. Run the Streamlit UI

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### 4. (Optional) Run the Retrieval Backend

The FastAPI backend for ingredient retrieval can be started with:

```bash
uvicorn src.backend.retrieval_server:app --reload --port 8000
```

The UI's `NUTRIGRAPH_BACKEND_URL` should point to this URL when backend integration is enabled.

## Configuration

Environment variables (set in `.env` or system environment):

| Variable | Default | Description |
|----------|---------|-------------|
| `NUTRIGRAPH_BACKEND_URL` | `http://localhost:8000` | Backend API URL |
| `NUTRIGRAPH_ENV` | `local` | Environment (local/staging) |

## Development

### Current Status

- Streamlit UI is functional but still uses **mock nutrition estimates**.
- A FastAPI **retrieval backend** exists for ingredient ↔ vector DB lookups.

### Mock Mode

UI-level API calls currently return mock data:
- Nutrition estimates are generated from dish names (deterministic based on name)
- Ingredients are randomly generated but reproducible
- No actual backend connection is required

### Future Integration

The `NutriGraphClient` class in `src/core/api_client.py` contains placeholder methods ready for backend integration:
- `estimate_nutrition()` - Will call RAG-powered estimation endpoint
- `builder_generate_profile()` - Will call ingredient-based calculation endpoint

## Tech Stack

- **Streamlit** - Web application framework
- **Pydantic** - Data validation and settings management
- **Pandas** - Data manipulation for tables and exports
- **python-dotenv** - Environment variable management

## License

Course project - for educational purposes.
