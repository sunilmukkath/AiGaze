import Image from "next/image";

/** Approximate aspect from elastic-tree-logo.png (426×77). */
const LOGO_ASPECT = 426 / 77;

export function ElasticTreeLogo({
  height = 28,
  className = "",
  priority = false,
}: {
  height?: number;
  className?: string;
  priority?: boolean;
}) {
  const width = Math.round(height * LOGO_ASPECT);
  return (
    <Image
      src="/elastic-tree-logo.png"
      alt="Elastic Tree"
      width={width}
      height={height}
      className={`shrink-0 object-contain ${className}`}
      priority={priority}
      style={{ height, width: "auto", maxWidth: width }}
    />
  );
}
