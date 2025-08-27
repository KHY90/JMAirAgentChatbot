import os
from supabase.client import Client, create_client
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL과 Key는 환경 변수에 설정해야 합니다.")

supabase: Client = create_client(supabase_url, supabase_key)
