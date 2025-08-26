
import os
from supabase.client import Client, create_client
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Supabase URL and Key must be set in the environment variables.")

supabase: Client = create_client(supabase_url, supabase_key)
