/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://54.210.154.126.nip.io/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
