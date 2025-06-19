import os
from bs4 import BeautifulSoup

NAV_HTML = """
            <ul class="menu">
                <li><a href="/"><i class="fas fa-home"></i> Home</a></li>
                <li><a href="/getting-started.html"><i class="fas fa-rocket"></i> Getting Started</a></li>
                <li class="has-submenu">
                    <a href="#"><i class="fas fa-book"></i> Documentation <i class="fas fa-chevron-down"></i></a>
                    <ul class="submenu">
                        <li><a href="/developer-guide.html">Developer Guide</a></li>
                        <li><a href="/dsl-specification.html">DSL Specification</a></li>
                        <li><a href="/faq.html">FAQ</a></li>
                    </ul>
                </li>
                <li class="has-submenu">
                    <a href="#"><i class="fas fa-star"></i> Features <i class="fas fa-chevron-down"></i></a>
                    <ul class="submenu">
                        <li><a href="/dsl-specification.html#lifecycle-nodes">ROS2 Lifecycle</a></li>
                        <li><a href="/dsl-specification.html#qos-configuration">QoS Configuration</a></li>
                        <li><a href="/dsl-specification.html#cuda-integration">CUDA Integration</a></li>
                    </ul>
                </li>
                <li class="has-submenu">
                    <a href="#"><i class="fas fa-users"></i> Community <i class="fas fa-chevron-down"></i></a>
                    <ul class="submenu">
                        <li><a href="/contributing.html">Contributing</a></li>
                        <li><a href="/code-of-conduct.html">Code of Conduct</a></li>
                    </ul>
                </li>
            </ul>
"""

def update_navigation(html_file):
    with open(html_file, 'r+', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        
        # Find the menu
        menu = soup.find('ul', class_='menu')
        if menu:
            # Create new navigation
            new_nav = BeautifulSoup(NAV_HTML, 'html.parser')
            menu.replace_with(new_nav)
            
            # Update active link
            current_page = os.path.basename(html_file)
            if current_page == 'index.html':
                current_page = '/'
            
            for a in new_nav.find_all('a', href=True):
                if a['href'] == current_page or \
                   (current_page != '/' and a['href'].endswith(current_page)):
                    a['class'] = 'active'
            
            # Write back to file
            f.seek(0)
            f.write(str(soup))
            f.truncate()
            print(f"Updated navigation in {html_file}")
        else:
            print(f"No menu found in {html_file}")

def main():
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.html'):
                update_navigation(os.path.join(root, file))

if __name__ == '__main__':
    main()
