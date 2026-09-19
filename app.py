from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "0.0.0.0"
PORT = 3000


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"Hello from a tiny Python app!\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"Serving on http://{HOST}:{PORT}")
    HTTPServer((HOST, PORT), Handler).serve_forever()
