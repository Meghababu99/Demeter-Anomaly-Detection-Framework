import pandas as pd
from datetime import timedelta
from shapely.geometry import Polygon, Point

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np

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
        # shapely expects (x, y) = (lon, lat)
        return Polygon(list(zip(longitudes, latitudes)))

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
            eq_point = Point(eq['long'], eq['lat'])  # (lon, lat)
            eq_data = eq[['lat', 'long', 'Time', 'mag']].to_dict()

            if anomaly_polygon.contains(eq_point):
                inside_spatial_window.append(eq_data)
                any_inside = True
            else:
                outside_spatial_window.append(eq_data)

        label = 1 if any_inside else 0
        return label, inside_spatial_window, outside_spatial_window

    def label_sequences(self, dataset, earthquakes, save_path=None):
        """
        Build sequence labels for a HalfOrbitPairDataset by checking if any point in each
        combined half-orbit sequence is seismic within the configured time window.

        Args:
            dataset: HalfOrbitPairDataset instance (e.g., train/val/test dataset).
            earthquakes (pd.DataFrame): Earthquake catalog with columns ['Time','lat','long','mag'].
            save_path (str, optional): If provided, CSV path to save the summary.

        Returns:
            pd.DataFrame with columns ['index', 'time_window', 'label']
        """
        # Use a criteria instance configured like this object
        criteria = SeismicCriteria(self.spatial_width, self.time_window_hours)

        # Build sequences (uses the original df the dataset was built from)
        _, dt_seqs, latlon_seqs, _ = dataset.create_half_orbit_sequences(dataset.df)

        sequence_summary = []
        total_seismic_sequences = 0
        matched_eqs_records = []  

        for idx, (dt_seq, latlon_seq) in enumerate(zip(dt_seqs, latlon_seqs)):
            # last timestamp of the combined sequence
            anomaly_time = dt_seq[-1]

            # Short-circuit as soon as one location is seismic
            label = 0
            seq_matched_eqs = [] 
            for loc in latlon_seq:
                # loc expected as [lat, lon] or (lat, lon)
                is_seismic, inside_eqs, _ = criteria.is_eq(anomaly_time, loc, earthquakes)
                if inside_eqs:
                    seq_matched_eqs.extend(inside_eqs)  # add all matching eqs

                if is_seismic:  # at least one inside
                    label = 1
            sequence_summary.append({
                "index": idx,
                "time_window": anomaly_time,
                "label": label,
                "matched_eqs": seq_matched_eqs,
                'total_eq':len(seq_matched_eqs)

            })
            matched_eqs_records.extend(seq_matched_eqs)

        summary_df = pd.DataFrame(sequence_summary).reset_index(drop=True)
        if matched_eqs_records:
            matched_eqs_df = pd.DataFrame(matched_eqs_records).drop_duplicates()
            # matched_eqs_df.to_csv(save_path, index=False)

        else:
            matched_eqs_df = pd.DataFrame(columns=["lat","long","Time","mag"])
        # Save if requested
        if save_path:
            summary_df.to_csv(save_path, index=False)

            print(f"[INFO] Saved summary DataFrame to {save_path}")

        print(f"[INFO] Total seismic sequences found: {total_seismic_sequences}")
        print(f"[INFO] Unique earthquakes matched: {len(matched_eqs_df)}")
        return summary_df
    




class SeismicCriteria_half_orbit: # Used for half orbit datas as well as the 23_20 SW
    def __init__(self, spatial_width_a=None, spatial_width_b=None, time_window_hours=None):
        """
        Initialize criteria for detecting if an anomaly is associated with an earthquake.

        Args:
            spatial_width_a (float): East-west width (longitude) in degrees.
            spatial_width_b (float): North-south height (latitude) in degrees.
            time_window_hours (int): Time window in hours after the anomaly timestamp.
        """
        self.spatial_width_a = spatial_width_a
        self.spatial_width_b = spatial_width_b
        self.time_window_hours = time_window_hours

    def create_polygon(self, lat, lon):
        """
        Creates a rectangular polygon around a given latitude and longitude
        with different horizontal and vertical spans.

        Args:
            lat (float): Latitude center of the anomaly.
            lon (float): Longitude center of the anomaly.

        Returns:
            shapely.geometry.Polygon: Rectangle centered at (lat, lon).
        """
        # Half widths
        delta_lon = self.spatial_width_a / 2   # horizontal (E-W)
        delta_lat = self.spatial_width_b / 2   # vertical (N-S)

        # Rectangle vertices
        latitudes = [lat - delta_lat, lat + delta_lat, lat + delta_lat, lat - delta_lat]
        longitudes = [lon - delta_lon, lon - delta_lon, lon + delta_lon, lon + delta_lon]

        return Polygon(list(zip(longitudes, latitudes)))


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
        # anomaly_polygon  =build_orbit_polygon(anomaly_location)

        inside_spatial_window = []
        outside_spatial_window = []
        any_inside = False

        for _, eq in filtered_eq.iterrows():
            eq_point = Point(eq['long'], eq['lat'])  # (lon, lat)
            eq_data = eq[['lat', 'long', 'Time', 'mag']].to_dict()

            if anomaly_polygon.contains(eq_point):
                inside_spatial_window.append(eq_data)
                any_inside = True
            else:
                outside_spatial_window.append(eq_data)

        label = 1 if any_inside else 0
        return label, inside_spatial_window, outside_spatial_window

  
    # =========================================================
    # 1️⃣ FUNCTION: build DEMETER orbit polygon
    # =========================================================
 


    def label_sequences(self, dataset, earthquakes, save_path=None):
        """
        Build sequence labels for a HalfOrbitPairDataset by checking if any point in each
        combined half-orbit sequence is seismic within the configured time window.

        Args:
            dataset: HalfOrbitPairDataset instance (e.g., train/val/test dataset).
            earthquakes (pd.DataFrame): Earthquake catalog with columns ['Time','lat','long','mag'].
            save_path (str, optional): If provided, CSV path to save the summary.

        Returns:
            pd.DataFrame with columns ['index', 'time_window', 'label']
        """
        # Use a criteria instance configured like this object
        criteria = SeismicCriteria_half_orbit(self.spatial_width_a, self.spatial_width_b, self.time_window_hours)


        # Build sequences (uses the original df the dataset was built from)
        _, dt_seqs, latlon_seqs, _ = dataset.create_half_orbit_sequences(dataset.df)

        sequence_summary = []
        total_seismic_sequences = 0
        matched_eqs_records = []  

        for idx, (dt_seq, latlon_seq) in enumerate(zip(dt_seqs, latlon_seqs)):
            # last timestamp of the combined sequence
            anomaly_time = dt_seq[-1]

            # Short-circuit as soon as one location is seismic
            label = 0
            seq_matched_eqs = [] 
            for loc in latlon_seq:
                    
                is_seismic, inside_eqs, _ = criteria.is_eq(anomaly_time, loc, earthquakes)
                if inside_eqs:
                    seq_matched_eqs.extend(inside_eqs)  # add all matching eqs

                if is_seismic:  # at least one inside
                    label = 1
            sequence_summary.append({
                "index": idx,
                "time_window": anomaly_time,
                "label": label,
                "matched_eqs": seq_matched_eqs,
                'total_eq':len(seq_matched_eqs)

            })
            matched_eqs_records.extend(seq_matched_eqs)

        summary_df = pd.DataFrame(sequence_summary).reset_index(drop=True)
        if matched_eqs_records:
            matched_eqs_df = pd.DataFrame(matched_eqs_records).drop_duplicates()
            # matched_eqs_df.to_csv(save_path, index=False)

        else:
            matched_eqs_df = pd.DataFrame(columns=["lat","long","Time","mag"])
        # Save if requested
        if save_path:
            summary_df.to_csv(save_path, index=False)

            print(f"[INFO] Saved summary DataFrame to {save_path}")

        print(f"[INFO] Total seismic sequences found: {total_seismic_sequences}")
        print(f"[INFO] Unique earthquakes matched: {len(matched_eqs_df)}")
        return summary_df

