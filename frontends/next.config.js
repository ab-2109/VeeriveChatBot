/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://54.210.154.126:8000/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
