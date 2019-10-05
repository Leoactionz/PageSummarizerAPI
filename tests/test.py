from sure import expect
import requests
import httpretty
import unittest


class RestTest(unittest.TestCase):
    @httpretty.activate
    # Testing GET Request
    def test_one(self):
        httpretty.register_uri(httpretty.GET, "http://127.0.0.1:5000/",
                               body="{\"message\": \"This is SMS spam detection model. Use the format {'message': 'SMS message'} and POST to get prediction.\"}")

        response = requests.get('http://127.0.0.1:5000/')

        print(response)
        assert response.text == "{\"message\": \"This is SMS spam detection model. Use the format {'message': 'SMS message'} and POST to get prediction.\"}"

    # Testing POST Request
    def test_two(self):
        httpretty.register_uri(
            httpretty.POST, "http://127.0.0.1:5000/")

        response = requests.post('http://127.0.0.1:5000/',
                                 headers={'Content-Type': 'application/json'}, data='{"message": "Welcome home"}')
        assert str(response.json()
                   ) == "{'results': 'Not a spam'}"


if __name__ == '__main__':
    unittest.main()
