"""
Deduplicate template matching detections to ensure each event appears only once.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

from .base import BaseProcessor

console = Console()


class DetectionDeduplicator(BaseProcessor):
    """Deduplicate overlapping detections from multiple templates."""
    
    def cluster_detections_by_time(
        self, 
        df: pd.DataFrame, 
        time_window_sec: float = 5.0
    ) -> pd.DataFrame:
        """
        Cluster detections that are close in time and likely represent the same event.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: time, template_id, similarity, event_amplitude, etc.
        time_window_sec : float
            Maximum time difference (seconds) for detections to be considered the same event.
            
        Returns
        -------
        pd.DataFrame
            Deduplicated DataFrame with one row per event.
        """
        
        if len(df) == 0:
            return df
        
        # Convert times to Unix timestamps for clustering
        df = df.copy()
        df['unix_time'] = df['time'].astype(np.int64) // 10**9
        
        # Use DBSCAN to cluster by time
        X = df['unix_time'].values.reshape(-1, 1)
        
        # Convert time window to DBSCAN eps (seconds)
        eps = time_window_sec
        
        clustering = DBSCAN(
            eps=eps,
            min_samples=1,
            metric='euclidean'
        ).fit(X)
        
        df['time_cluster'] = clustering.labels_
        
        return df
    
    def merge_cluster_detections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge multiple detections within the same time cluster.
        
        Strategy: Keep the detection with highest similarity, or compute weighted average.
        """
        
        merged_rows = []
        
        for cluster_id, cluster_df in df.groupby('time_cluster'):
            if len(cluster_df) == 1:
                # Single detection - keep as is
                row = cluster_df.iloc[0].copy()
                row['n_duplicate_detections'] = 1
                row['duplicate_template_ids'] = str([int(row['template_id'])])
                merged_rows.append(row)
            else:
                # Multiple detections of the same event
                # Choose the one with highest similarity
                best_idx = cluster_df['similarity'].idxmax()
                best_row = cluster_df.loc[best_idx].copy()
                
                # Store information about duplicates
                best_row['n_duplicate_detections'] = len(cluster_df)
                best_row['duplicate_template_ids'] = str(cluster_df['template_id'].astype(int).tolist())
                
                # Optionally compute weighted averages for amplitude and magnitude
                if 'event_amplitude' in cluster_df.columns:
                    weights = cluster_df['similarity'].values
                    weighted_amplitude = np.average(
                        cluster_df['event_amplitude'].values, 
                        weights=weights
                    )
                    best_row['event_amplitude'] = weighted_amplitude
                
                if 'est_magnitude' in cluster_df.columns and cluster_df['est_magnitude'].notna().any():
                    valid_mags = cluster_df['est_magnitude'].dropna()
                    if len(valid_mags) > 0:
                        weights = cluster_df.loc[valid_mags.index, 'similarity'].values
                        weighted_mag = np.average(valid_mags.values, weights=weights)
                        best_row['est_magnitude'] = weighted_mag
                
                merged_rows.append(best_row)
        
        merged_df = pd.DataFrame(merged_rows)
        
        # Drop helper columns
        columns_to_drop = ['unix_time', 'time_cluster']
        merged_df = merged_df.drop(columns=[c for c in columns_to_drop if c in merged_df.columns])
        
        return merged_df
    
    def deduplicate_detections(
        self, 
        df: pd.DataFrame,
        time_window_sec: float = 5.0,
        use_spatial_clustering: bool = False,
        spatial_window_km: float = 10.0
    ) -> pd.DataFrame:
        """
        Main deduplication workflow.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input detections DataFrame
        time_window_sec : float
            Time window for considering detections as duplicates (seconds)
        use_spatial_clustering : bool
            If True, also cluster by location (requires lat/lon columns)
        spatial_window_km : float
            Spatial window for clustering (kilometers)
            
        Returns
        -------
        pd.DataFrame
            Deduplicated DataFrame
        """
        
        console.print(f"[cyan]🔄 Deduplicating {len(df)} detections...[/cyan]")
        
        if len(df) == 0:
            return df
        
        # 1. Sort by time
        df_sorted = df.sort_values('time').reset_index(drop=True)
        
        # 2. Cluster by time
        df_clustered = self.cluster_detections_by_time(df_sorted, time_window_sec)
        
        # 3. If we have location information, refine clusters
        if use_spatial_clustering and 'lat' in df_clustered.columns and 'lon' in df_clustered.columns:
            df_clustered = self._refine_clusters_with_location(
                df_clustered, 
                spatial_window_km
            )
        
        # 4. Merge clusters
        df_deduplicated = self.merge_cluster_detections(df_clustered)
        
        # 5. Add unique event ID
        df_deduplicated = df_deduplicated.sort_values('time').reset_index(drop=True)
        df_deduplicated['event_id'] = np.arange(1, len(df_deduplicated) + 1)
        
        console.print(f"✅ Deduplication complete: {len(df)} → {len(df_deduplicated)} events")
        console.print(f"   Removed {len(df) - len(df_deduplicated)} duplicate detections")
        
        # Statistics
        if 'n_duplicate_detections' in df_deduplicated.columns:
            n_multiples = (df_deduplicated['n_duplicate_detections'] > 1).sum()
            console.print(f"   {n_multiples} events detected by multiple templates")
        
        return df_deduplicated
    
    def _refine_clusters_with_location(
        self, 
        df: pd.DataFrame, 
        spatial_window_km: float
    ) -> pd.DataFrame:
        """
        Refine time-based clusters using spatial information.
        
        Converts lat/lon to approximate km and applies additional clustering.
        """
        
        from obspy.geodetics import degrees2kilometers
        
        # Create refined cluster IDs
        refined_clusters = []
        
        for time_cluster, cluster_df in df.groupby('time_cluster'):
            if len(cluster_df) == 1:
                refined_clusters.append((time_cluster, cluster_df, [0]))
                continue
            
            # Compute pairwise distances in km
            lats = cluster_df['lat'].values
            lons = cluster_df['lon'].values
            
            # Approximate km per degree at this latitude
            km_per_deg_lat = 111.32
            km_per_deg_lon = 111.32 * np.cos(np.radians(np.mean(lats)))
            
            # Convert to km coordinates
            x_km = (lons - lons.mean()) * km_per_deg_lon
            y_km = (lats - lats.mean()) * km_per_deg_lat
            
            # Use DBSCAN with spatial window
            X = np.column_stack([x_km, y_km])
            clustering = DBSCAN(
                eps=spatial_window_km,
                min_samples=1
            ).fit(X)
            
            refined_clusters.append((time_cluster, cluster_df, clustering.labels_))
        
        # Reassign cluster IDs
        df_refined = pd.DataFrame()
        new_cluster_id = 0
        
        for time_cluster, cluster_df, sub_labels in refined_clusters:
            for sub_label in np.unique(sub_labels):
                mask = (sub_labels == sub_label)
                sub_cluster = cluster_df[mask].copy()
                sub_cluster['refined_cluster'] = new_cluster_id
                df_refined = pd.concat([df_refined, sub_cluster])
                new_cluster_id += 1
        
        # Replace time_cluster with refined_cluster
        df_refined = df_refined.rename(columns={'refined_cluster': 'time_cluster'})
        
        return df_refined.reset_index(drop=True)