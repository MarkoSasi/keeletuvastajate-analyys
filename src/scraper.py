import time
import random
import requests
from typing import Optional, List
from config import USER_AGENTS, REQUEST_DELAY, SUBREDDIT, SUBREDDITS


class RedditScraper:

    BASE_URL = "https://www.reddit.com"
    MAX_RETRIES = 3

    def __init__(self, proxies: Optional[List[str]] = None):
        """
        Initsialiseeri koguja valikulise proksitoetusega.

        Args:
            proxies: Proksi-URL-ide loend (nt ["http://proxy1:8080", "socks5://proxy2:1080"])
        """
        self.session = requests.Session()
        self.proxies = proxies or []
        self.current_proxy_index = 0
        self.consecutive_errors = 0
        self._update_user_agent()

    def _update_user_agent(self):
        """Rotateeru user-agentti, et vältida tuvastamist."""
        self.session.headers.update({
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9,et;q=0.8",
        })

    def _get_proxy(self) -> Optional[dict]:
        """Tagasta praegune proksikonfiguratsioon."""
        if not self.proxies:
            return None

        proxy_url = self.proxies[self.current_proxy_index % len(self.proxies)]
        return {"http": proxy_url, "https": proxy_url}

    def _rotate_proxy(self):
        """Vali loendis järgmine proksile."""
        if self.proxies:
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
            print(f"  Rotating to proxy {self.current_proxy_index + 1}/{len(self.proxies)}")

    def _make_request(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        

        for attempt in range(self.MAX_RETRIES):
            try:
                # Lisa viivitusele juhuslik kõikumine (kasvab järjestikuste vigadega)
                base_delay = REQUEST_DELAY * (1 + self.consecutive_errors * 0.5)
                delay = base_delay + random.uniform(0, 2)
                time.sleep(delay)

                # Roteeri user-agentti aeg-ajalt
                if random.random() < 0.3:
                    self._update_user_agent()

                # Tee päring valikulise proksiga
                proxy = self._get_proxy()
                response = self.session.get(
                    url,
                    params=params,
                    timeout=30,
                    proxies=proxy
                )

                # Päringute piirang
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    print(f"  Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                # Keelatud staatus (IP blokeeritud)
                if response.status_code == 403:
                    self.consecutive_errors += 1
                    print(f"  403 Forbidden (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    if self.proxies:
                        self._rotate_proxy()
                    else:
                        # Eksponentsiaalne ootamine ilma proksita
                        wait_time = 30 * (2 ** attempt)
                        print(f"  Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    continue

                response.raise_for_status()
                self.consecutive_errors = max(0, self.consecutive_errors - 1)  # Vähenda vigade arvu õnnestumise korral
                return response.json()
            
            except requests.exceptions.Timeout:
                print(f"  Timeout (attempt {attempt + 1}/{self.MAX_RETRIES})")
                continue
            
            except requests.exceptions.ConnectionError as e:
                print(f"  Connection error: {e}")
                if self.proxies:
                    self._rotate_proxy()
                continue
            
            except requests.exceptions.RequestException as e:
                print(f"  Request error: {e}")
                return None
            
            except ValueError as e:
                print(f"  JSON decode error: {e}")
                return None
        
        print(f"  Failed after {self.MAX_RETRIES} attempts")
        return None
    
    def fetch_subreddit_posts(
        self,
        category: str = "hot",
        limit: int = 100,
        time_filter: str = "all",
        after: Optional[str] = None,
        subreddit: Optional[str] = None
    ) -> list[dict]:
        """
        Tõmba subredditist postitusi.

        Args:
            category: hot, new, top, rising, controversial
            limit: Tõmmatavate postituste arv (max 100 päringu kohta)
            time_filter: top/controversial korral: hour, day, week, month, year, all
            after: Lehe märk järgmise lehe jaoks
            subreddit: Sihtsubreddit (vaikimisi config.SUBREDDIT)

        Returns:
            Postitusi kirjeldavate sõnastike loend
        """
        sub = subreddit or SUBREDDIT
        url = f"{self.BASE_URL}/r/{sub}/{category}.json"
        params = {"limit": min(limit, 100), "raw_json": 1}
        
        if category in ["top", "controversial"]:
            params["t"] = time_filter
        
        if after:
            params["after"] = after
        
        data = self._make_request(url, params)
        if not data:
            return []
        
        posts = []
        try:
            for child in data.get("data", {}).get("children", []):
                post_data = child.get("data", {})
                posts.append({
                    "id": post_data.get("id", ""),
                    "title": post_data.get("title", ""),
                    "selftext": post_data.get("selftext", ""),
                    "permalink": post_data.get("permalink", ""),
                    "score": post_data.get("score", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "url": f"{self.BASE_URL}{post_data.get('permalink', '')}",
                    "after": data.get("data", {}).get("after"),
                })
        except (KeyError, TypeError) as e:
            print(f"Error parsing posts: {e}")
        
        return posts
    
    def fetch_post_comments(self, permalink: str, limit: int = 500) -> list[str]:
        """
        Tõmba postituse kõik kommentaarid.

        Args:
            permalink: Postituse permalink (nt /r/Eesti/comments/abc123/title/)
            limit: Maksimaalne tõmmatavate kommentaaride arv

        Returns:
            Kommentaaride loend
        """
        url = f"{self.BASE_URL}{permalink}.json"
        params = {"limit": limit, "raw_json": 1}
        
        data = self._make_request(url, params)
        if not data or len(data) < 2:
            return []
        
        comments = []
        self._extract_comments(data[1].get("data", {}).get("children", []), comments)
        return comments
    
    def _extract_comments(self, children: list, comments: list):
        """Eralda rekursiivselt kommentaaride tekst."""
        for child in children:
            if child.get("kind") != "t1":
                continue

            data = child.get("data", {})
            body = data.get("body", "")

            # Jäta vahele kustutatud/eemaldatud kommentaarid
            if body and body not in ["[deleted]", "[removed]"]:
                comments.append(body)

            # Töötle vastused rekursiivselt
            replies = data.get("replies")
            if isinstance(replies, dict):
                reply_children = replies.get("data", {}).get("children", [])
                self._extract_comments(reply_children, comments)

    def search_subreddit(self, query: str, limit: int = 100, subreddit: Optional[str] = None) -> list[dict]:
        """
        Otsi subredditist postitusi.

        Args:
            query: Otsingupäring stringina
            limit: Maksimaalne tulemuste arv
            subreddit: Otsitav subreddit (vaikimisi config.SUBREDDIT)

        Returns:
            Postitusi kirjeldavate sõnastike loend
        """
        sub = subreddit or SUBREDDIT
        url = f"{self.BASE_URL}/r/{sub}/search.json"
        params = {
            "q": query,
            "restrict_sr": "on",
            "limit": min(limit, 100),
            "raw_json": 1,
            "sort": "relevance",
        }
        
        data = self._make_request(url, params)
        if not data:
            return []
        
        posts = []
        try:
            for child in data.get("data", {}).get("children", []):
                post_data = child.get("data", {})
                posts.append({
                    "id": post_data.get("id", ""),
                    "title": post_data.get("title", ""),
                    "selftext": post_data.get("selftext", ""),
                    "permalink": post_data.get("permalink", ""),
                    "score": post_data.get("score", 0),
                    "url": f"{self.BASE_URL}{post_data.get('permalink', '')}",
                })
        except (KeyError, TypeError) as e:
            print(f"Error parsing search results: {e}")
        
        return posts
    
    def fetch_all_posts(self, category: str = "new", max_posts: int = 500, time_filter: str = "all", subreddit: Optional[str] = None) -> list[dict]:
        """
        Tõmba mitu lehte postitusi.

        Args:
            category: Postituse kategooria
            max_posts: Maksimaalne tõmmatavate postituste koguarv
            time_filter: Ajafilter top/controversial jaoks
            subreddit: Sihtsubreddit (vaikimisi config.SUBREDDIT)

        Returns:
            Kõigi tõmmatud postituste loend
        """
        all_posts = []
        after = None
        
        while len(all_posts) < max_posts:
            remaining = max_posts - len(all_posts)
            limit = min(100, remaining)
            
            posts = self.fetch_subreddit_posts(
                category=category,
                limit=limit,
                time_filter=time_filter,
                after=after,
                subreddit=subreddit
            )
            
            if not posts:
                break
            
            all_posts.extend(posts)
            after = posts[-1].get("after") if posts else None
            
            if not after:
                break
            
            print(f"  Fetched {len(all_posts)} posts so far...")
        
        return all_posts


if __name__ == "__main__":
    scraper = RedditScraper()
    
    print("Testing subreddit fetch...")
    posts = scraper.fetch_subreddit_posts(category="hot", limit=5)
    print(f"Fetched {len(posts)} posts")
    
    if posts:
        print(f"\nFirst post title: {posts[0]['title'][:50]}...")
        print(f"\nFetching comments for first post...")
        comments = scraper.fetch_post_comments(posts[0]['permalink'])
        print(f"Fetched {len(comments)} comments")
