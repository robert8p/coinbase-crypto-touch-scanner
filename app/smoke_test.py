import os
from starlette.testclient import TestClient
from app.main import app

def main():
    c = TestClient(app)
    assert c.get("/health").status_code==200
    assert c.get("/").status_code==200
    st = c.get("/api/status").json()
    assert "coinbase" in st
    sc = c.get("/api/scores?limit=1").json()
    assert "rows" in sc
    print("smoke ok")

if __name__=="__main__":
    main()
