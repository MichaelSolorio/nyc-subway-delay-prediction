"""
MTA GTFS-Realtime Data Collection
Fetches live subway data from the MTA API for L, 6, and A trains

HOW THIS WORKS:
1. We make an HTTP request to MTA's API with our API key
2. MTA returns data in "Protocol Buffer" format (a compressed binary format)
3. We use gtfs-realtime-bindings to decode it into Python objects
4. We extract delay information and save it for each route
"""

import requests
from google.transit import gtfs_realtime_pb2
from datetime import datetime
import json
import os
import sys

# Add project root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config


def fetch_mta_data(route_key):
    """
    Fetch real-time GTFS data from MTA API for a specific route

    Args:
        route_key: Route key ('L', '6', or 'A')

    Returns:
        gtfs_realtime_pb2.FeedMessage: Parsed GTFS feed, or None if error
    """
    route_config = config.MTA_FEEDS.get(route_key)
    if not route_config:
        print(f"[ERROR] Unknown route: {route_key}")
        return None

    print(f"[FETCH] Fetching MTA GTFS data for {route_config['name']}...")

    # Set up the request (API key optional - MTA made feeds public)
    headers = {}
    if config.MTA_API_KEY and config.MTA_API_KEY != 'PASTE_YOUR_MTA_KEY_HERE':
        headers['x-api-key'] = config.MTA_API_KEY

    try:
        # Make the HTTP request to MTA
        response = requests.get(route_config['feed_url'], headers=headers)

        # Check if request was successful
        if response.status_code != 200:
            print(f"[ERROR] MTA API returned status {response.status_code}")
            if response.status_code == 401:
                print("   Your API key may be invalid")
            elif response.status_code == 403:
                print("   Access forbidden - check your API key")
            return None

        # Parse the Protocol Buffer response
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(response.content)

        print(f"[SUCCESS] Successfully fetched data!")
        print(f"   Feed timestamp: {datetime.fromtimestamp(feed.header.timestamp)}")
        print(f"   Number of entities: {len(feed.entity)}")

        return feed

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network error: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Error parsing feed: {e}")
        return None


def parse_trip_updates(feed, route_id):
    """
    Extract delay information from the GTFS feed for a specific route

    Args:
        feed: GTFS-realtime FeedMessage
        route_id: Route ID to filter for ('L', '6', 'A')

    Returns:
        list: List of dictionaries with delay information
    """
    if feed is None:
        return []

    delays = []

    for entity in feed.entity:
        # Check if this entity has trip update info
        if entity.HasField('trip_update'):
            trip = entity.trip_update

            # Get trip info
            trip_id = trip.trip.trip_id
            entity_route_id = trip.trip.route_id

            # Only process the specified route
            if entity_route_id != route_id:
                continue

            # Go through each stop time update
            for stop_update in trip.stop_time_update:
                delay_data = {
                    'trip_id': trip_id,
                    'route_id': entity_route_id,
                    'stop_id': stop_update.stop_id,
                    'timestamp': datetime.now().isoformat(),
                }

                # Check for arrival delay
                if stop_update.HasField('arrival'):
                    delay_data['arrival_delay_seconds'] = stop_update.arrival.delay
                    delay_data['arrival_time'] = stop_update.arrival.time

                # Check for departure delay
                if stop_update.HasField('departure'):
                    delay_data['departure_delay_seconds'] = stop_update.departure.delay
                    delay_data['departure_time'] = stop_update.departure.time

                delays.append(delay_data)

    return delays


def parse_vehicle_positions(feed, route_id):
    """
    Extract current train positions from the feed for a specific route

    Args:
        feed: GTFS-realtime FeedMessage
        route_id: Route ID to filter for ('L', '6', 'A')

    Returns:
        list: List of dictionaries with vehicle positions
    """
    if feed is None:
        return []

    positions = []

    for entity in feed.entity:
        if entity.HasField('vehicle'):
            vehicle = entity.vehicle

            # Only process the specified route
            if vehicle.trip.route_id != route_id:
                continue

            position_data = {
                'trip_id': vehicle.trip.trip_id,
                'route_id': vehicle.trip.route_id,
                'current_stop_id': vehicle.stop_id,
                'current_status': vehicle.current_status,  # 0=incoming, 1=stopped, 2=in_transit
                'timestamp': datetime.fromtimestamp(vehicle.timestamp).isoformat() if vehicle.timestamp else None,
            }

            positions.append(position_data)

    return positions


def parse_alerts(feed, route_id):
    """
    Extract service alerts from the feed for a specific route

    Args:
        feed: GTFS-realtime FeedMessage
        route_id: Route ID to filter for ('L', '6', 'A')

    Returns:
        list: List of alert dictionaries
    """
    if feed is None:
        return []

    alerts = []

    for entity in feed.entity:
        if entity.HasField('alert'):
            alert = entity.alert

            # Check if alert affects this route
            affects_route = False
            for informed in alert.informed_entity:
                if informed.route_id == route_id:
                    affects_route = True
                    break

            if not affects_route:
                continue

            alert_data = {
                'alert_id': entity.id,
                'route_id': route_id,
                'header': alert.header_text.translation[0].text if alert.header_text.translation else '',
                'description': alert.description_text.translation[0].text if alert.description_text.translation else '',
                'timestamp': datetime.now().isoformat(),
            }

            alerts.append(alert_data)

    return alerts


def save_raw_data(data, route_id, data_type='delays'):
    """
    Save raw data to data/raw/ folder as JSON

    Args:
        data: Data to save (list of dictionaries)
        route_id: Route ID ('L', '6', 'A')
        data_type: Type of data ('delays', 'positions', 'alerts')
    """
    # Create data directory if it doesn't exist
    # Go from src/data_collection/ up to project root, then into data/raw/
    raw_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    raw_path = os.path.abspath(raw_path)  # Clean up the path
    os.makedirs(raw_path, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{route_id}_train_{data_type}_{timestamp}.json"
    filepath = os.path.join(raw_path, filename)

    # Save as JSON
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[SAVED] {len(data)} records to: {filename}")
    return filepath


def collect_all_data():
    """
    Main function: Fetch and save all MTA data for all routes

    Returns:
        dict: Summary of collected data
    """
    print("=" * 60)
    print("ATLAS - MTA Data Collection (L, 6, A Trains)")
    print("=" * 60)
    print()

    total_summary = {
        'L': {'delays': 0, 'positions': 0, 'alerts': 0},
        '6': {'delays': 0, 'positions': 0, 'alerts': 0},
        'A': {'delays': 0, 'positions': 0, 'alerts': 0}
    }

    # Collect data for each route
    for route_key in config.SUPPORTED_ROUTES:
        route_config = config.MTA_FEEDS[route_key]
        print(f"\n[ROUTE] Collecting data for {route_config['name']}")
        print("-" * 60)
        print()

        # Fetch the feed
        feed = fetch_mta_data(route_key)

        if feed is None:
            print(f"[ERROR] Data collection failed for {route_config['name']}!")
            continue

        print()

        # Parse different data types
        print("[PARSE] Parsing trip updates (delays)...")
        delays = parse_trip_updates(feed, route_key)
        print(f"   Found {len(delays)} delay records")
        total_summary[route_key]['delays'] = len(delays)

        print("[PARSE] Parsing vehicle positions...")
        positions = parse_vehicle_positions(feed, route_key)
        print(f"   Found {len(positions)} vehicle positions")
        total_summary[route_key]['positions'] = len(positions)

        print("[PARSE] Parsing service alerts...")
        alerts = parse_alerts(feed, route_key)
        print(f"   Found {len(alerts)} alerts")
        total_summary[route_key]['alerts'] = len(alerts)

        print()

        # Save the data
        if delays:
            save_raw_data(delays, route_key, 'delays')
        if positions:
            save_raw_data(positions, route_key, 'positions')
        if alerts:
            save_raw_data(alerts, route_key, 'alerts')

    print()
    print("=" * 60)
    print("[DONE] Data collection complete for all routes!")
    print("=" * 60)
    print("\nSummary:")
    for route_key in config.SUPPORTED_ROUTES:
        route_name = config.MTA_FEEDS[route_key]['name']
        stats = total_summary[route_key]
        print(f"  {route_name}: {stats['delays']} delays, {stats['positions']} positions, {stats['alerts']} alerts")

    return total_summary


# This runs when you execute this file directly
if __name__ == "__main__":
    collect_all_data()
