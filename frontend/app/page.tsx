"use client";

import { motion } from "framer-motion";
import {
  Brain,
  FileText,
  Users,
  Zap,
  ArrowRight,
  Layers,
  MessageSquare,
  Hand,
  Sparkles,
  Lock,
  Github,
  ChevronRight,
} from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PERSONAS, type PersonaId } from "@/lib/types";

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 },
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1,
    },
  },
};

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 glass border-b border-border/50">
        <div className="container mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-nexus-gradient flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold">Nexus</span>
          </Link>
          <div className="hidden md:flex items-center gap-8">
            <Link
              href="#features"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              Features
            </Link>
            <Link
              href="#personas"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              AI Personas
            </Link>
            <Link
              href="#how-it-works"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              How It Works
            </Link>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/workspace">
              <Button variant="glow">
                Launch App
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6 relative overflow-hidden">
        {/* Background Effects */}
        <div className="absolute inset-0 bg-gradient-radial from-primary/10 via-transparent to-transparent" />
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-nexus-blue/20 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-nexus-purple/20 rounded-full blur-3xl" />

        <motion.div
          className="container mx-auto text-center relative z-10"
          initial="initial"
          animate="animate"
          variants={staggerContainer}
        >
          <motion.div variants={fadeInUp}>
            <Badge variant="info" className="mb-6">
              <Sparkles className="w-3 h-3 mr-1" />
              Powered by RAPTOR Retrieval
            </Badge>
          </motion.div>

          <motion.h1
            className="text-5xl md:text-7xl font-bold mb-6 leading-tight"
            variants={fadeInUp}
          >
            Your Personal Team of
            <br />
            <span className="gradient-text">AI Research Specialists</span>
          </motion.h1>

          <motion.p
            className="text-xl text-muted-foreground max-w-3xl mx-auto mb-8"
            variants={fadeInUp}
          >
            Transform document chaos into an intelligent, collaborative AI
            workspace. Organize 1000s of documents, query with human-like
            understanding, and delegate work to specialist AI personas that know
            when to ask for your help.
          </motion.p>

          <motion.div
            className="flex flex-col sm:flex-row gap-4 justify-center"
            variants={fadeInUp}
          >
            <Link href="/workspace">
              <Button size="xl" variant="gradient" className="group">
                Get Started Free
                <ChevronRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
            <Button size="xl" variant="outline">
              <Github className="w-5 h-5 mr-2" />
              View on GitHub
            </Button>
          </motion.div>

          {/* Hero Visual */}
          <motion.div
            className="mt-16 relative"
            variants={fadeInUp}
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8 }}
          >
            <div className="relative mx-auto max-w-5xl">
              <div className="absolute inset-0 bg-nexus-gradient rounded-2xl blur-3xl opacity-20" />
              <div className="relative glass rounded-2xl border border-border/50 p-2 shadow-2xl">
                <div className="bg-background rounded-xl overflow-hidden">
                  {/* Mock Canvas Preview */}
                  <div className="aspect-video bg-gradient-to-br from-background to-card relative">
                    {/* Mock Nodes */}
                    <div className="absolute top-8 left-8 w-48 h-32 glass rounded-xl border border-persona-max/30 p-4 animate-float">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-2xl">🔧</span>
                        <span className="font-semibold text-sm">
                          Mechanics
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        12 documents
                      </p>
                      <div className="mt-2 flex gap-1">
                        <Badge variant="outline" className="text-xs">
                          CAD
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          Motors
                        </Badge>
                      </div>
                    </div>

                    <div
                      className="absolute top-20 right-16 w-48 h-32 glass rounded-xl border border-persona-elena/30 p-4 animate-float"
                      style={{ animationDelay: "0.5s" }}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-2xl">⚛️</span>
                        <span className="font-semibold text-sm">Physics</span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        8 documents
                      </p>
                      <div className="mt-2 flex gap-1">
                        <Badge variant="outline" className="text-xs">
                          Theory
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          EM
                        </Badge>
                      </div>
                    </div>

                    <div
                      className="absolute bottom-16 left-1/3 w-48 h-32 glass rounded-xl border border-persona-byte/30 p-4 animate-float"
                      style={{ animationDelay: "1s" }}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-2xl">💻</span>
                        <span className="font-semibold text-sm">Code</span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        24 documents
                      </p>
                      <div className="mt-2 flex gap-1">
                        <Badge variant="outline" className="text-xs">
                          Python
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          ML
                        </Badge>
                      </div>
                    </div>

                    {/* Connection Lines (SVG) */}
                    <svg className="absolute inset-0 w-full h-full pointer-events-none">
                      <defs>
                        <linearGradient
                          id="lineGradient"
                          x1="0%"
                          y1="0%"
                          x2="100%"
                          y2="0%"
                        >
                          <stop offset="0%" stopColor="#3B82F6" />
                          <stop offset="100%" stopColor="#8B5CF6" />
                        </linearGradient>
                      </defs>
                      <path
                        d="M 200 100 Q 400 150 500 120"
                        stroke="url(#lineGradient)"
                        strokeWidth="2"
                        fill="none"
                        opacity="0.5"
                      />
                      <path
                        d="M 200 130 Q 350 200 400 250"
                        stroke="url(#lineGradient)"
                        strokeWidth="2"
                        fill="none"
                        opacity="0.5"
                      />
                    </svg>

                    {/* Chat Preview */}
                    <div className="absolute bottom-4 right-4 w-72 glass rounded-xl border border-border/50 p-3">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-6 h-6 rounded-full bg-persona-max flex items-center justify-center text-xs">
                          🔧
                        </div>
                        <span className="text-sm font-medium">Max</span>
                        <Badge variant="success" className="text-xs ml-auto">
                          Online
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        &quot;Hey! I found the torque specs you needed. The motor can
                        handle 5G acceleration with a 3x safety margin...&quot;
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-6">
        <div className="container mx-auto">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <Badge variant="purple" className="mb-4">
              Features
            </Badge>
            <h2 className="text-4xl font-bold mb-4">
              Everything you need for intelligent research
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Nexus combines cutting-edge retrieval technology with an intuitive
              interface to transform how you work with documents.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                icon: Layers,
                title: "RAPTOR Hierarchical Retrieval",
                description:
                  "Multi-layer tree structure captures both high-level themes and low-level details. Ask summary questions or dive deep—Nexus adapts.",
                color: "text-nexus-blue",
              },
              {
                icon: Users,
                title: "Specialist AI Personas",
                description:
                  "Work with Max (Mechanical), Dr. Elena (Physics), Byte (Code), and more. Each specialist has unique expertise and personality.",
                color: "text-nexus-purple",
              },
              {
                icon: Hand,
                title: "Human-in-the-Loop",
                description:
                  "Agents know their limits and request your help with structured tasks, safety guidance, and step-by-step instructions.",
                color: "text-nexus-orange",
              },
              {
                icon: FileText,
                title: "Infinite Canvas",
                description:
                  "Organize documents visually on a 2D canvas. Create groups, draw connections, and build your knowledge mind-map.",
                color: "text-nexus-cyan",
              },
              {
                icon: Zap,
                title: "Lightning Fast & Local",
                description:
                  "Process 1000s of documents locally. Your data never leaves your machine—privacy by design.",
                color: "text-nexus-green",
              },
              {
                icon: MessageSquare,
                title: "Natural Conversations",
                description:
                  'Talk naturally: "Hey Max, will this motor handle 5G acceleration?" Get responses with citations and source links.',
                color: "text-nexus-pink",
              },
            ].map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="h-full hover:border-primary/50 transition-colors group">
                  <CardContent className="p-6">
                    <div
                      className={`w-12 h-12 rounded-xl bg-card flex items-center justify-center mb-4 group-hover:scale-110 transition-transform ${feature.color}`}
                    >
                      <feature.icon className="w-6 h-6" />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-muted-foreground text-sm">
                      {feature.description}
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Personas Section */}
      <section id="personas" className="py-20 px-6 bg-card/50">
        <div className="container mx-auto">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <Badge variant="info" className="mb-4">
              Meet Your Team
            </Badge>
            <h2 className="text-4xl font-bold mb-4">
              AI Specialists with Personality
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Each persona has unique expertise, communication style, and
              quirks. They collaborate across domains and know when to ask for
              your help.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {(Object.keys(PERSONAS) as PersonaId[]).map((id, index) => {
              const persona = PERSONAS[id];
              return (
                <motion.div
                  key={id}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card
                    className="h-full overflow-hidden group hover:shadow-lg transition-all"
                    style={{
                      borderColor: `${persona.color}30`,
                    }}
                  >
                    <CardContent className="p-6">
                      <div
                        className="w-16 h-16 rounded-2xl flex items-center justify-center text-3xl mb-4 group-hover:scale-110 transition-transform"
                        style={{ backgroundColor: `${persona.color}20` }}
                      >
                        {persona.avatar}
                      </div>
                      <h3 className="text-lg font-semibold mb-1">
                        {persona.name}
                      </h3>
                      <p
                        className="text-sm mb-3"
                        style={{ color: persona.color }}
                      >
                        {persona.role}
                      </p>
                      <p className="text-muted-foreground text-sm mb-4">
                        {persona.description}
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {persona.traits.map((trait) => (
                          <Badge key={trait} variant="outline" className="text-xs">
                            {trait}
                          </Badge>
                        ))}
                      </div>
                      <div
                        className="mt-4 p-3 rounded-lg text-sm italic"
                        style={{ backgroundColor: `${persona.color}10` }}
                      >
                        &quot;{persona.greeting}&quot;
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 px-6">
        <div className="container mx-auto">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <Badge variant="success" className="mb-4">
              How It Works
            </Badge>
            <h2 className="text-4xl font-bold mb-4">
              From documents to insights in minutes
            </h2>
          </motion.div>

          <div className="max-w-4xl mx-auto">
            {[
              {
                step: "01",
                title: "Upload Your Documents",
                description:
                  "Drag and drop PDFs, DOCX, or TXT files. Nexus automatically parses, chunks, and embeds your content using semantic analysis.",
              },
              {
                step: "02",
                title: "Organize on Canvas",
                description:
                  "Create document groups, assign specialists, and draw connections. Your visual layout becomes your knowledge map.",
              },
              {
                step: "03",
                title: "Build RAPTOR Tree",
                description:
                  "Nexus clusters your chunks and generates hierarchical summaries. Layer 0 has details, higher layers have themes.",
              },
              {
                step: "04",
                title: "Query Your Specialists",
                description:
                  'Ask natural questions: "Hey Elena, explain the physics behind this." Get answers with citations and source links.',
              },
            ].map((item, index) => (
              <motion.div
                key={item.step}
                className="flex gap-6 mb-12 last:mb-0"
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="flex-shrink-0">
                  <div className="w-12 h-12 rounded-full bg-nexus-gradient flex items-center justify-center text-white font-bold">
                    {item.step}
                  </div>
                  {index < 3 && (
                    <div className="w-px h-16 bg-border mx-auto mt-4" />
                  )}
                </div>
                <div className="pt-2">
                  <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                  <p className="text-muted-foreground">{item.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6">
        <div className="container mx-auto">
          <motion.div
            className="relative overflow-hidden rounded-3xl bg-nexus-gradient p-12 text-center"
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
          >
            <div className="absolute inset-0 bg-black/20" />
            <div className="relative z-10">
              <Lock className="w-12 h-12 mx-auto mb-6 text-white/80" />
              <h2 className="text-4xl font-bold text-white mb-4">
                Local-First. Private by Design.
              </h2>
              <p className="text-white/80 max-w-2xl mx-auto mb-8">
                All processing happens on your machine. Your documents never
                leave your device. Nexus is the AI research assistant that
                respects your privacy.
              </p>
              <Link href="/workspace">
                <Button
                  size="xl"
                  variant="secondary"
                  className="bg-white text-primary hover:bg-white/90"
                >
                  Start Building Your Knowledge Base
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-border">
        <div className="container mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-nexus-gradient flex items-center justify-center">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <span className="font-semibold">Nexus</span>
            </div>
            <p className="text-muted-foreground text-sm">
              © 2026 Nexus. Open source under MIT License.
            </p>
            <div className="flex gap-4">
              <Link
                href="#"
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                <Github className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
