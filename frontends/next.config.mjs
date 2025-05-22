/** @type {import('next').NextConfig} */
const nextConfig = {
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: 'http://<YOUR-EC2-PUBLIC-IP>:8000/:path*',
        },
      ];
    },
  };
  
  module.exports = nextConfig;
  