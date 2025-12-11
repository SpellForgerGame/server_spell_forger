import requests
import json
import time
from statistics import mean, median, stdev

# Server configuration
SERVER_URL = "http://127.0.0.1:8000"
ENDPOINT = "/generate-spell/"

def test_generate_spell(description):
    """
    Test function to call the generate-spell endpoint
    Returns tuple: (result, response_time_seconds)
    """
    url = f"{SERVER_URL}{ENDPOINT}"
    
    # Prepare the request payload
    payload = {
        "description": description
    }
    
    try:
        # Start timer
        start_time = time.time()
        
        # Make the POST request
        response = requests.post(url, json=payload)
        
        # End timer
        end_time = time.time()
        response_time = end_time - start_time
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! (Response time: {response_time:.3f}s)")
            print(f"Input: {description}")
            print(f"Result: {result}")
            return result, response_time
        else:
            print(f"❌ Error: {response.status_code} (Response time: {response_time:.3f}s)")
            print(f"Response: {response.text}")
            return None, response_time
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server. Make sure the server is running.")
        return None, 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None, 0

def test_multiple_spells():
    """
    Test function to call the endpoint with multiple spell descriptions
    Returns list of response times
    """
    test_descriptions = [
        "A fireball spell that deals massive damage",
        "A healing spell that restores health over time", 
        "An ice spell that freezes enemies in place",
        "A lightning bolt that strikes multiple targets",
        "A shield spell that protects from magical attacks"
    ]
    
    print("🧙‍♂️ Testing multiple spell generations...\n")
    
    response_times = []
    successful_requests = 0
    
    for i, description in enumerate(test_descriptions, 1):
        print(f"--- Test {i}/{len(test_descriptions)} ---")
        result, response_time = test_generate_spell(description)
        if result is not None:
            response_times.append(response_time)
            successful_requests += 1
        print()
    
    return response_times, successful_requests

def performance_benchmark(num_tests=10):
    """
    Run multiple tests to benchmark server performance
    """
    print(f"🚀 Running performance benchmark with {num_tests} tests...")
    print("=" * 50)
    
    test_descriptions = [
        "A fireball spell that deals massive damage",
        "A healing spell that restores health over time", 
        "An ice spell that freezes enemies in place",
        "A lightning bolt that strikes multiple targets",
        "A shield spell that protects from magical attacks",
        "A teleportation spell that moves the caster instantly",
        "A poison spell that damages over time",
        "A wind spell that pushes enemies away",
        "A earth spell that creates stone barriers",
        "A water spell that extinguishes fires"
    ]
    
    response_times = []
    successful_requests = 0
    
    for i in range(num_tests):
        # Use different descriptions cyclically
        description = test_descriptions[i % len(test_descriptions)]
        
        print(f"Test {i+1}/{num_tests}: {description[:50]}...")
        
        result, response_time = test_generate_spell(description)
        if result is not None:
            response_times.append(response_time)
            successful_requests += 1
        
        # Small delay between requests to avoid overwhelming the server
        time.sleep(0.1)
    
    return response_times, successful_requests

def display_performance_stats(response_times, successful_requests, total_requests):
    """
    Display detailed performance statistics
    """
    if not response_times:
        print("❌ No successful requests to analyze")
        return
    
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE STATISTICS")
    print("=" * 50)
    
    # Basic stats
    avg_time = mean(response_times)
    median_time = median(response_times)
    min_time = min(response_times)
    max_time = max(response_times)
    
    # Standard deviation (only if we have more than 1 sample)
    std_dev = stdev(response_times) if len(response_times) > 1 else 0
    
    # Success rate
    success_rate = (successful_requests / total_requests) * 100
    
    print(f"📈 Success Rate: {success_rate:.1f}% ({successful_requests}/{total_requests})")
    print(f"⚡ Average Response Time: {avg_time:.3f} seconds")
    print(f"📊 Median Response Time: {median_time:.3f} seconds")
    print(f"🚀 Fastest Response: {min_time:.3f} seconds")
    print(f"🐌 Slowest Response: {max_time:.3f} seconds")
    print(f"📏 Standard Deviation: {std_dev:.3f} seconds")
    
    # Performance categories
    print(f"\n🎯 PERFORMANCE BREAKDOWN:")
    fast_requests = sum(1 for t in response_times if t < 1.0)
    medium_requests = sum(1 for t in response_times if 1.0 <= t < 3.0)
    slow_requests = sum(1 for t in response_times if t >= 3.0)
    
    print(f"   ⚡ Fast (< 1s): {fast_requests} requests ({fast_requests/len(response_times)*100:.1f}%)")
    print(f"   ⏱️  Medium (1-3s): {medium_requests} requests ({medium_requests/len(response_times)*100:.1f}%)")
    print(f"   🐌 Slow (> 3s): {slow_requests} requests ({slow_requests/len(response_times)*100:.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if avg_time < 1.0:
        print("   ✅ Excellent performance! Response times are very fast.")
    elif avg_time < 3.0:
        print("   ⚠️  Good performance, but there might be room for optimization.")
    else:
        print("   ❌ Slow performance. Consider optimizing your model or server.")
    
    if std_dev > avg_time * 0.5:
        print("   ⚠️  High variability in response times. Consider investigating inconsistent performance.")

def check_server_health():
    """
    Check if the server is running and accessible
    """
    try:
        start_time = time.time()
        response = requests.get(f"{SERVER_URL}/docs")
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Server is running and accessible (Health check: {response_time:.3f}s)")
            return True
        else:
            print(f"⚠️  Server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not accessible. Make sure it's running on http://127.0.0.1:8000")
        return False

if __name__ == "__main__":
    print("🔮 SpellForge Server Test Client")
    print("=" * 40)
    
    # Check if server is running
    if not check_server_health():
        print("\n💡 To start the server, run: python main.py")
        exit(1)
    
    print()
    
    # Test single spell generation
    print("🧪 Testing single spell generation...")
    result, response_time = test_generate_spell("A powerful fire spell that burns enemies")
    
    print("\n" + "=" * 40)
    
    # Test multiple spells
    response_times, successful_requests = test_multiple_spells()
    
    if response_times:
        display_performance_stats(response_times, successful_requests, 5)
    
    print("\n" + "=" * 40)
    
    # Ask user if they want to run benchmark
    user_input = input("\n🚀 Would you like to run a performance benchmark? (y/n): ").lower().strip()
    if user_input in ['y', 'yes']:
        num_tests = input("How many tests? (default: 10): ").strip()
        try:
            num_tests = int(num_tests) if num_tests else 10
        except ValueError:
            num_tests = 10
        
        benchmark_times, benchmark_successful = performance_benchmark(num_tests)
        display_performance_stats(benchmark_times, benchmark_successful, num_tests)
    
    print("\n🎉 Testing completed!")
