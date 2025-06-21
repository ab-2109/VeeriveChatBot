/** @type {import('next').NextConfig} */
const nextConfig = {
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: 'http://3.86.52.25.nip.io/:path*',
        },
      ];
    },
  };
  
module.exports = nextConfig;
  