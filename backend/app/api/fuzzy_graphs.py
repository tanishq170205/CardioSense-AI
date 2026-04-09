from fastapi import APIRouter
from app.fuzzy import get_fuzzy_plot_data

router = APIRouter()

@router.get("/fuzzy_graphs")
async def fetch_fuzzy_graphs():
    """
    Returns the X, Y coordinate data for rendering all the fuzzy membership function graphs.
    """
    data = get_fuzzy_plot_data()
    return data
