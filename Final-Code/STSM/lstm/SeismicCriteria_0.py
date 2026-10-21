import pandas as pd
from datetime import timedelta
from shapely.geometry import Point, Polygon

class SeismicCriteria:
    def __init__(self, spatial_width=None, time_window_hours=None):
        """
        Initialize criteria for detecting if an anomaly is associated with an earthquake.

        Args:
            spatial_width (float): Width (in degrees) of the spatial window around anomaly location.
            time_window_hours (int): Time window in hours after the anomaly timestamp.
        """
        self.spatial_width = spatial_width
        self.time_window_hours = time_window_hours

    def create_polygon(self, lat, lon):
        """
        Creates a square polygon around a given latitude and longitude.

        Args:
            lat (float): Latitude of the anomaly.
            lon (float): Longitude of the anomaly.

        Returns:
            shapely.geometry.Polygon: Square polygon centered at (lat, lon).
        """
        delta = self.spatial_width / 2
        latitudes = [lat - delta, lat + delta, lat + delta, lat - delta]
        longitudes = [lon - delta, lon - delta, lon + delta, lon + delta]
        return Polygon(zip(longitudes, latitudes))

    def is_eq(self, timestamp, anomaly_location, earthquakes):
        """
        Determines if an earthquake occurred within the spatial and temporal window of an anomaly.

        Args:
            timestamp (str or datetime): Time of the anomaly.
            anomaly_location (tuple): (lat, lon) of the anomaly.
            earthquakes (DataFrame): DataFrame of earthquakes with 'Time', 'lat', 'long', 'mag'.

        Returns:
            tuple:
                - int: 1 if any earthquake is within the window, else 0.
                - list of dicts: Earthquakes inside the window.
                - list of dicts: Earthquakes outside the window but within time.
        """
        timestamp = pd.to_datetime(timestamp)
        start_time = timestamp
        end_time = timestamp + timedelta(hours=self.time_window_hours)

        filtered_eq = earthquakes[
            (earthquakes['Time'] >= start_time) &
            (earthquakes['Time'] <= end_time)
        ]

        lat, lon = anomaly_location
        anomaly_polygon = self.create_polygon(lat, lon)

        inside_spatial_window = []
        outside_spatial_window = []
        any_inside = False

        for _, eq in filtered_eq.iterrows():
            eq_point = Point(eq['long'], eq['lat'])
            eq_data = eq[['lat', 'long', 'Time', 'mag']].to_dict()

            if anomaly_polygon.contains(eq_point):
                inside_spatial_window.append(eq_data)
                any_inside = True
            else:
                outside_spatial_window.append(eq_data)

        label = 1 if any_inside else 0
        return label, inside_spatial_window, outside_spatial_window
