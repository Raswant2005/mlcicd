from fastapi import FastAPI ,Path, HTTPException
import json

app=FastAPI()


def get_data():
    with open(r"C:\Users\raswa\OneDrive\Desktop\Docker\data\raw\patients.json") as f:
        data = json.load(f)
    return data

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items") 
def read_item():
    return {"This is Item API":"DOOMSDAY"}

@app.get("/items/view")
def view_data():
    data = get_data()
    return data 

@app.get("/items/view/{items_id}")
def display_data(items_id:str = Path(..., title="The ID of the item to get", min_length=1, max_length=10)):
    data = get_data() 
    
    if items_id in data:
        return data[items_id]
    raise HTTPException(status_code=404, detail="Item not found")