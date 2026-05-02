import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Script from "next/script";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "LLM VRAM Calculator",
  description:
    "Estimate whether a given language model will fit in your GPU's VRAM for inference, LoRA, or QLoRA fine-tuning.",
  openGraph: {
    title: "LLM VRAM Calculator",
    description:
      "Estimate whether a given language model will fit in your GPU's VRAM for inference, LoRA, or QLoRA fine-tuning.",
    type: "website",
  },
};

const themeInitScript = `
(function() {
  try {
    var stored = localStorage.getItem('theme');
    var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (stored === 'dark' || (!stored && prefersDark)) {
      document.documentElement.classList.add('dark');
    }
  } catch (e) {}
})();
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} antialiased`}
      suppressHydrationWarning
    >
      <head>
        <Script id="theme-init" strategy="beforeInteractive">
          {themeInitScript}
        </Script>
      </head>
      <body className="min-h-screen flex flex-col bg-paper text-graphite dark:bg-ink dark:text-stone">
        <main className="flex-1 w-full">{children}</main>
      </body>
    </html>
  );
}
