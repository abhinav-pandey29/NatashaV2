# Natasha

### ⚙️ Configure Your Environment

Before running the project, ensure your environment is correctly set up. This involves adding the required environment variables and generating the necessary credentials from Spotify.

#### 1️⃣ Set Up Environment Variables

Create a `.env` file in the project root and add the following variables:

```ini
SPOTIFY_APPLICATION_PATH=/path/to/spotify_app
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=your_spotify_redirect_uri
SPOTIFY_USER_ID=your_spotify_user_id
```

#### 2️⃣ Generate Spotify API Credentials

To obtain your `SPOTIPY_CLIENT_ID` and `SPOTIPY_CLIENT_SECRET`:

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/applications).
2. Log in and create a new application.
3. Copy the **Client ID** and **Client Secret**.
4. Set the **Redirect URI** to `http://localhost:8888/callback` in your app settings.

#### 3️⃣ You're All Set!

Once your `.env` file is ready, your environment is configured, and you can proceed with running the project. 🚀
