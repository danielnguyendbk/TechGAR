import { expect, test } from "@playwright/test";

test.describe.configure({ mode: "serial" });

test("capture required mobile and desktop screenshots", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.addStyleTag({ content: ".mock-toggle, .mock-panel { display: none !important; }" });
  await expect(page.getByRole("heading", { name: "Bạn muốn tìm chỗ đỗ theo cách nào?" })).toBeVisible();
  await page.screenshot({ path: "artifacts/screenshots/390x844-entry.png", fullPage: false });

  await page.getByTestId("entry-recommend").click();
  await page.getByTestId("need-entertainment").click();
  await expect(page.getByTestId("recommendation-result")).toBeVisible();
  await page.screenshot({ path: "artifacts/screenshots/390x844-recommendation.png", fullPage: false });

  await page.getByTestId("recommendation-confirm").click();
  await expect(page.getByTestId("active-route")).toBeVisible();
  await page.screenshot({ path: "artifacts/screenshots/390x844-navigation.png", fullPage: false });

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await page.addStyleTag({ content: ".mock-toggle, .mock-panel { display: none !important; }" });
  await page.getByTestId("entry-skip").click();
  await expect(page.locator("[data-spot-id]")).toHaveCount(160);
  await page.screenshot({ path: "artifacts/screenshots/1440x900-desktop.png", fullPage: false });

  await page.getByTestId("spot-C10").click();
  await page.getByTestId("spot-navigate").click();
  await expect(page.getByTestId("active-route")).toBeVisible();
  await page.screenshot({ path: "artifacts/screenshots/1440x900-c10-navigation.png", fullPage: false });
});
