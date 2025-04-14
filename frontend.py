import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import spotipy
import requests
import random
import time
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

# Initialize session state variables if they don't exist
if 'spotify_authenticated' not in st.session_state:
    st.session_state.spotify_authenticated = False
if 'spotify_client' not in st.session_state:
    st.session_state.spotify_client = None
if 'displayed_songs' not in st.session_state:
    st.session_state.displayed_songs = []
if 'selections' not in st.session_state:
    st.session_state.selections = []
if 'ratings' not in st.session_state:
    st.session_state.ratings = []
if 'user_liked_songs' not in st.session_state:
    st.session_state.user_liked_songs = []

# This would normally be populated from your database
song_IDs = []  # You can add some Spotify IDs here for testing

def authenticate_spotify():
    """Set up Spotify authentication for user login"""
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope='user-library-read user-top-read',
        cache_path='.spotify_cache'
    )
    
    # Check if we need to authenticate
    if not st.session_state.spotify_authenticated:
        # Get the authorization URL
        if 'auth_url' not in st.session_state:
            auth_url = auth_manager.get_authorize_url()
            st.session_state.auth_url = auth_url
        
        # Display login button
        st.markdown(f"[![Login with Spotify]"
               f"(https://img.shields.io/badge/Login%20with-Spotify-1DB954?style=for-the-badge&logo=spotify&logoColor=white)]"
               f"({st.session_state.auth_url})")
        
        # Get the authorization code from the URL after redirect
        auth_code = None
        if "code" in st.query_params:
            auth_code = st.query_params["code"]
            st.info("Detected redirect from Spotify. Processing authentication...")
        
        if auth_code:
            try:
                # Exchange code for access token
                auth_manager.get_access_token(auth_code)
                st.session_state.spotify_authenticated = True
                st.session_state.spotify_client = spotipy.Spotify(auth_manager=auth_manager)
                st.success("Successfully authenticated with Spotify!")
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
    
    return st.session_state.spotify_client if st.session_state.spotify_authenticated else None

def fetch_user_liked_songs(sp):
    """Fetch the user's liked/saved tracks"""
    if not sp:
        return []
    
    liked_songs = []
    results = sp.current_user_saved_tracks(limit=50)
    
    while results:
        for item in results['items']:
            track = item['track']
            liked_songs.append({
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None
            })
        
        if results['next']:
            results = sp.next(results)
        else:
            break
    
    return liked_songs

def fetch_random_songs_directly(sp, num_songs=20):
    """Fetch random songs directly without genre seeds"""
    if not sp:
        return []
    
    collected_songs = []
    # List of random search characters to get diverse results
    search_chars = list('abcdefghijklmnopqrstuvwxyz')
    
    while len(collected_songs) < num_songs and search_chars:
        # Pick a random character and remove it from the list to avoid duplicates
        char = random.choice(search_chars)
        search_chars.remove(char)
        
        try:
            # Add a random offset for more variety
            offset = random.randint(0, 500)
            results = sp.search(q=char, type='track', limit=100, offset=offset)
            
            if results and 'tracks' in results and 'items' in results['tracks']:
                for track in results['tracks']['items']:
                    # Skip tracks already in our collection
                    if track['id'] in [s['id'] for s in collected_songs]:
                        continue
                    
                    # Skip tracks without an album image
                    if not track['album']['images']:
                        continue
                    
                    collected_songs.append({
                        'id': track['id'],
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'image_url': track['album']['images'][0]['url']
                    })
                    
                    if len(collected_songs) >= num_songs:
                        break
            
            # Avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            st.warning(f"Error in random search with '{char}': {e}")
            continue
    
    return collected_songs

def fetch_recommendations_from_tracks(sp, tracks, num_songs=20):
    """Fetch recommendations based on seed tracks"""
    if not sp or not tracks:
        return []
    
    collected_songs = []
    
    # Use up to 5 tracks as seeds (Spotify limit)
    seed_tracks = random.sample(tracks, min(5, len(tracks)))
    seed_ids = [track['id'] for track in seed_tracks]
    
    try:
        results = sp.recommendations(seed_tracks=seed_ids, limit=num_songs)
        
        if results and 'tracks' in results:
            for track in results['tracks']:
                if not track['album']['images']:
                    continue
                
                collected_songs.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'image_url': track['album']['images'][0]['url']
                })
    except Exception as e:
        st.warning(f"Error getting recommendations from seed tracks: {e}")
    
    return collected_songs

def fetch_random_songs(sp, num_songs=20):
    """Fetch random songs from Spotify"""
    if not sp:
        return []
    
    # Get initial batch of songs directly
    initial_songs = fetch_random_songs_directly(sp, min(20, num_songs))
    
    if len(initial_songs) >= num_songs:
        return initial_songs[:num_songs]
    
    # If we don't have enough songs yet, try getting recommendations
    # based on the songs we already found
    if initial_songs:
        recommended_songs = fetch_recommendations_from_tracks(sp, initial_songs, num_songs - len(initial_songs))
        
        # Combine and filter out duplicates
        all_song_ids = [song['id'] for song in initial_songs]
        for song in recommended_songs:
            if song['id'] not in all_song_ids:
                initial_songs.append(song)
                all_song_ids.append(song['id'])
                
                if len(initial_songs) >= num_songs:
                    break
    
    # If we still don't have enough songs, get more direct searches
    if len(initial_songs) < num_songs:
        more_songs = fetch_random_songs_directly(sp, num_songs - len(initial_songs))
        
        # Add more songs, avoiding duplicates
        all_song_ids = [song['id'] for song in initial_songs]
        for song in more_songs:
            if song['id'] not in all_song_ids:
                initial_songs.append(song)
                
                if len(initial_songs) >= num_songs:
                    break
    
    return initial_songs

def get_song_image_from_url(image_url):
    """Get song image from URL or create a placeholder"""
    try:
        if image_url:
            response = requests.get(image_url)
            img = Image.open(io.BytesIO(response.content))
            img = img.resize((120, 120))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()
    except Exception:
        pass
    
    # Fallback to placeholder
    return create_song_image(hash(image_url) % 100 if image_url else random.randint(0, 100))

def create_song_image(index, size=(120, 120)):
    """Create a colored image with play button for a song"""
    # Generate color based on index
    r = (index * 30) % 256
    g = (index * 50) % 256
    b = (index * 70) % 256
    
    # Create image
    img = Image.new('RGB', size, color=(r, g, b))
    draw = ImageDraw.Draw(img)
    
    # Draw white play triangle
    draw.polygon([(45, 40), (45, 80), (85, 60)], fill="white")
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def main():
    st.title("Music Recommendation System")
    
    # Spotify authentication section
    sp = authenticate_spotify()
    
    if sp and st.session_state.spotify_authenticated:
        # Fetch user's liked songs if not already loaded
        if not st.session_state.user_liked_songs:
            with st.spinner("Fetching your liked songs..."):
                st.session_state.user_liked_songs = fetch_user_liked_songs(sp)
            st.success(f"Loaded {len(st.session_state.user_liked_songs)} songs from your Spotify library")
            
        # Fetch songs for display if not already loaded
        if not st.session_state.displayed_songs:
            with st.spinner("Fetching songs for recommendation..."):
                # First try to use user's liked songs for recommendations
                if st.session_state.user_liked_songs:
                    st.session_state.displayed_songs = fetch_recommendations_from_tracks(
                        sp, 
                        st.session_state.user_liked_songs,
                        num_songs=20
                    )
                
                # If that didn't work, or didn't return enough songs, add some random ones
                if len(st.session_state.displayed_songs) < 20:
                    remaining = 20 - len(st.session_state.displayed_songs)
                    st.session_state.displayed_songs.extend(
                        fetch_random_songs(sp, num_songs=remaining)
                    )
                
                # Initialize selection and rating arrays
                liked_song_ids = [song['id'] for song in st.session_state.user_liked_songs]
                st.session_state.selections = [song['id'] in liked_song_ids for song in st.session_state.displayed_songs]
                st.session_state.ratings = [6 if song['id'] in liked_song_ids else 0 for song in st.session_state.displayed_songs]
            
            st.success(f"Loaded {len(st.session_state.displayed_songs)} songs for recommendation")
    
    # Sidebar for recommendation method
    with st.sidebar:
        st.header("Recommendation Method")
        method = st.selectbox(
            "Choose a method:",
            ["Two Tower Deep Retrieval", "Collaborative Filtering", "Auto"]
        )
        
        # Get recommendations button
        recommend_button = st.button(
            "Get Recommendations", 
            use_container_width=True,
            type="primary"  # Makes it blue
        )
    
    # Main content - song grid with checkboxes and sliders
    st.subheader("Select and rate songs")
    
    # Create a 4-column grid for songs
    num_songs = min(20, len(st.session_state.displayed_songs))
    cols_per_row = 4
    
    # Create rows with 4 columns each
    for row in range(0, num_songs, cols_per_row):
        cols = st.columns(cols_per_row)
        
        for i, col in enumerate(cols):
            idx = row + i
            
            if idx < num_songs and idx < len(st.session_state.displayed_songs):
                song = st.session_state.displayed_songs[idx]
                
                with col:
                    # Song image and details
                    st.image(
                        get_song_image_from_url(song.get('image_url')),
                        caption=f"{song['name']}\n{song['artist']}",
                        use_container_width=True
                    )
                    
                    # Checkbox for selection
                    is_selected = st.checkbox(
                        "Select", 
                        key=f"select_{idx}",
                        value=st.session_state.selections[idx] if idx < len(st.session_state.selections) else False
                    )
                    
                    # Update session state
                    if idx < len(st.session_state.selections):
                        st.session_state.selections[idx] = is_selected
                    
                    # Slider for rating (only enabled if song is selected)
                    if is_selected:
                        rating = st.slider(
                            "Rating", 
                            min_value=0, 
                            max_value=10, 
                            value=st.session_state.ratings[idx] if idx < len(st.session_state.ratings) else 0,
                            key=f"rating_{idx}"
                        )
                        if idx < len(st.session_state.ratings):
                            st.session_state.ratings[idx] = rating
                    else:
                        # Show disabled slider
                        st.slider(
                            "Rating", 
                            min_value=0, 
                            max_value=10, 
                            value=0,
                            key=f"disabled_rating_{idx}",
                            disabled=True
                        )
                        if idx < len(st.session_state.ratings):
                            st.session_state.ratings[idx] = 0
    
    # Handle recommendation button click
    if recommend_button:
        selected_songs = []
        
        for i in range(min(num_songs, len(st.session_state.selections))):
            if i < len(st.session_state.selections) and st.session_state.selections[i]:
                selected_songs.append({
                    'id': st.session_state.displayed_songs[i]['id'],
                    'name': st.session_state.displayed_songs[i]['name'],
                    'artist': st.session_state.displayed_songs[i]['artist'],
                    'rating': st.session_state.ratings[i]
                })
        
        with st.sidebar:
            if not st.session_state.spotify_authenticated:
                st.error("Please authenticate with Spotify to get recommendations.")
            elif not selected_songs:
                st.error("Please select at least one song to get recommendations.")
            else:
                selected_songs_str = ", ".join([f"{s['name']} by {s['artist']} (Rating: {s['rating']})" for s in selected_songs[:3]])
                if len(selected_songs) > 3:
                    selected_songs_str += f" and {len(selected_songs)-3} more"
                    
                st.success(f"**Recommendations requested!**\n\nUsing method: {method}\n\nSelected songs: {selected_songs_str}")

if __name__ == "__main__":
    main()