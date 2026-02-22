import json
import random
import os
from datetime import datetime, timezone, timedelta

def get_random_timestamp(peak_hour_bias=False):
    now = datetime.now(timezone.utc)
    random_days = random.randint(0, 30)
    
    if peak_hour_bias and random.random() < 0.6:
        # Bias towards morning rush (7-9 AM) or evening rush (6-9 PM)
        random_hours = random.choice([7, 8, 9, 18, 19, 20, 21])
    else:
        random_hours = random.randint(0, 23)
        
    random_minutes = random.randint(0, 59)
    past_date = now - timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
    return past_date.strftime("%Y-%m-%dT%H:%M:%SZ")

def add_noise(text):
    """Simulate human typing errors, slang, and varied punctuation to prevent monotonicity"""
    if random.random() < 0.15:
        # Lowercase everything (lazy typing)
        text = text.lower()
    
    if random.random() < 0.1:
        # ALL CAPS (angry typing)
        text = text.upper()
        
    if random.random() < 0.2:
        # Add excessive punctuation
        text += random.choice(["!!!!", "???", "!!?!", "..", "...."])
        
    if random.random() < 0.1:
        # Add common slang/filler at start
        text = random.choice(["Bro ", "Hey, ", "Pls fix: ", "Urgent!! ", "Admin, "]) + text
        
    # Simple typo simulation (swap adjacent characters)
    if random.random() < 0.05 and len(text) > 10:
        idx = random.randint(0, len(text) - 2)
        text = text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
        
    return text

def get_diverse_categories():
    return {
        "electricity": {
            "templates": [
                "Power outage in {location}, we need to study!",
                "The socket near the bed is sparking and smells like burning plastic.",
                "Ceiling fan is making a weird grinding noise in {room}.",
                "Voltage keeps dropping, my laptop charger stopped working.",
                "No electricity since 6 AM, how are we supposed to get ready?",
                "The tube light in the corridor is flickering constantly.",
                "Switchboard is loose and wires are exposed near {location}."
            ],
            "labels": ["High", "Critical", "Medium", "High", "Medium", "Low", "Critical"]
        },
        "food_mess": {
            "templates": [
                "The paneer served tonight was completely stale and sour.",
                "Found a dead cockroach in the dal. Disgusting.",
                "Plates in the mess are greasy and not washed properly.",
                "Milk served at breakfast was curdled.",
                "Why is the food always running out before 1 PM? Didn't get lunch.",
                "The drinking water cooler in {location} is dispensing muddy water.",
                "Mess staff not wearing hairnets or gloves."
            ],
            "labels": ["High", "Critical", "Medium", "High", "Medium", "Critical", "Medium"]
        },
        "cleaning_hygiene": {
            "templates": [
                "Washroom 3 on {location} is completely flooded. Smells terrible.",
                "Room {room} hasn't been swept for 4 days.",
                "Dustbins in the {location} are overflowing with garbage.",
                "Dead rat found near the staircase of {location}.",
                "The washing machines are filthy from inside.",
                "No toilet paper or handwash in the common bathroom.",
                "Black mold growing on the ceiling of shower stall 2."
            ],
            "labels": ["High", "Low", "Medium", "Critical", "Medium", "Medium", "High"]
        },
        "injury_safety": {
            "templates": [
                "My roommate slipped on the wet floor near {location} and twisted their ankle, send a kit!",
                "Main gate lock is broken, saw strangers entering at 2 AM.",
                "Broken glass window in {room}, cold air is blowing in.",
                "Fire alarm keeps randomly triggering on {location}.",
                "Urgent medical help needed in room {room}, someone fainted.",
                "The balcony railing in {location} is very loose, totally unsafe.",
                "A stray dog entered {location} and is acting aggressive."
            ],
            "labels": ["Critical", "Critical", "High", "High", "Critical", "Critical", "High"]
        },
        "maintenance_infrastructure": {
            "templates": [
                "Water heater in {location} isn't working, water is freezing.",
                "AC in room {room} is just blowing warm air.",
                "My bed frame is completely broken and collapsed.",
                "The lift in {location} got stuck between 3rd and 4th floor.",
                "Wardrobe lock is jammed, can't access my clothes.",
                "Ceiling is leaking water onto my bed in room {room}.",
                "The door handle to room {room} fell off."
            ],
            "labels": ["Medium", "Medium", "High", "Critical", "Low", "Critical", "Medium"]
        },
        "internet_wifi": {
            "templates": [
                "WiFi is completely dead in {location} since last night.",
                "Getting 1000ms ping, impossible to attend online classes.",
                "Router in {location} is blinking red.",
                "My device keeps disconnecting from eduroam every 5 mins.",
                "LAN port in room {room} is physically broken."
            ],
            "labels": ["High", "Medium", "Medium", "Low", "Low"]
        },
        "noise_disturbance": {
            "templates": [
                "People in room {room} are playing loud music at 3 AM.",
                "Construction noise outside {location} is unbearable during exam week.",
                "Dogs barking non-stop near the {location} entrance.",
                "Someone is playing basketball in the corridor of {location}."
            ],
            "labels": ["Medium", "Medium", "Low", "Low"]
        },
        "administrative": {
            "templates": [
                "I was double-charged for the mess fee this month.",
                "Warden is not signing my out-pass request.",
                "My room allocation is wrong on the portal.",
                "Laundry service lost two of my shirts."
            ],
            "labels": ["Medium", "Low", "Medium", "Low"]
        }
    }

def generate_complaints(num_samples=2000):
    hostels = ["H-Alpha", "H-Beta", "H-Gamma", "Block-A", "Block-B", "Tower-1"]
    rooms = [f"{floor}{room:02d}" for floor in range(1, 9) for room in range(1, 40)]
    locations = hostels + ["North Wing", "South Corridor", "Main Mess", "Library Night Canteen", "Basement Parking"]
    
    categories = get_diverse_categories()
    complaints = []
    
    for i in range(1, num_samples + 1):
        # Weight categories (maintenance and cleaning are usually more common than injuries)
        cats = list(categories.keys())
        cat_weights = [0.15, 0.15, 0.20, 0.05, 0.25, 0.10, 0.05, 0.05]
        cat_name = random.choices(cats, weights=cat_weights, k=1)[0]
        
        cat_data = categories[cat_name]
        
        template_idx = random.randint(0, len(cat_data["templates"]) - 1)
        template = cat_data["templates"][template_idx]
        base_label = cat_data["labels"][template_idx]
        
        # Simulate human inconsistency (10% chance to over/under react)
        label = base_label
        if random.random() < 0.1:
            label = random.choice(["Low", "Medium", "High", "Critical"])
            
        # Fill in template vars
        raw_text = template.format(
            location=random.choice(locations),
            room=random.choice(rooms)
        )
        
        # Add realistic noise/human elements
        final_text = add_noise(raw_text)
        
        # Determine if it was reported during peak hours (e.g. food complaints spike near meals)
        bias_peak = cat_name in ["food_mess", "maintenance_infrastructure"]
        
        record = {
            "id": f"C_{i:05d}",
            "text": final_text,
            "timestamp": get_random_timestamp(peak_hour_bias=bias_peak),
            "label": label,
            "metadata": {
                "hostel_id": random.choice(hostels),
                "room": random.choice(rooms),
                "category": cat_name,
                "reported_via": random.choice(["app", "portal", "kiosk", "app", "app"])
            }
        }
        
        complaints.append(record)

    # Sort sequentially by timestamp to simulate real data collection
    complaints.sort(key=lambda x: x["timestamp"])

    os.makedirs("data/raw/complaints", exist_ok=True)
    
    output_path = "data/raw/complaints/mock_training_data.json"
    with open(output_path, "w") as f:
        json.dump(complaints, f, indent=2)
        
    print(f"✅ Successfully generated {num_samples} highly diverse, non-monotonic complaints!")
    print(f"📊 Categories included: {', '.join(categories.keys())}")
    print(f"📁 Saved to: {output_path}")

if __name__ == "__main__":
    print("Initializing state-of-the-art mock grievance generator...")
    generate_complaints(10000)
    print("Done!")
